import os
from flask import Flask, request
import agentscope
import json
from agentscope.message import Msg
from agentscope.parsers.json_object_parser import MarkdownJsonObjectParser
from agentscope.models import ModelResponse

app = Flask(__name__)

api_key = "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"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def json_extract(expect_key, json_dict) -> str:
    sub_dict = {expect_key: json_dict.parsed[expect_key]}
    sub_json = json.dumps(sub_dict)
    return sub_json


def convert_string(input_string):
    # 使用split方法按照下划线分割字符串
    split_string = input_string.split('_')
    # 使用join方法将分割后的字符串列表用空格连接
    result_string = ' '.join(split_string)
    return result_string


def msg_concat(**kwargs):
    Concat = ""
    for key, value in kwargs.items():
        Concat += convert_string(key) + ': ' + value.content + '\n'
    concat_msg = Msg(name="assistant", content=Concat)
    return concat_msg


def generate_prompt(query, eg_in, eg_out):
    model_configs_path = os.path.join(BASE_DIR, "configs", "model_configs.json")
    agent_configs_path = os.path.join(BASE_DIR, "configs", "agent_configs.json")

    # Init all the agents!
    PromptGEN_group = agentscope.init(
        model_configs=model_configs_path,
        agent_configs=agent_configs_path,
        logger_level="DEBUG"
    )
    # user = UserAgent()
    Analyzer = PromptGEN_group[0]
    JsonFormatter = PromptGEN_group[1]
    KnowledgeRefiner = PromptGEN_group[2]
    IniPromptGenerator_eg = PromptGEN_group[3]
    IniPromptGenerator = PromptGEN_group[4]
    ValidationGenerator = PromptGEN_group[5]
    QAGenerator = PromptGEN_group[6]
    Evaluator = PromptGEN_group[7]
    PromptRevisor = PromptGEN_group[8]

    # Init a parser
    json_parser = MarkdownJsonObjectParser()

    # Work Flow:
    # Begin to input.
    # query = input("What kind of prompts do you want to generate: ")
    # eg_in = input("Please provide an example intput: ")
    # eg_out = input("Please provide an example output: ")

    # Embed inputs into the Msg.
    Query = Msg(name="User", content=query)
    Eg_input = Msg(name="Example_input", content=eg_in)
    Eg_output = Msg(name="Example_output", content=eg_out)

    # Step1: Analyze the user's requirements.
    Ana_unpar_out = Analyzer(Query)

    # Step2: Parse the str into '''json''' format.
    Json_unpar_out = JsonFormatter(Ana_unpar_out)

    # Step3: Re-format the '''json''' into real JSON object.
    Json_pure_out = json_parser.parse(
        ModelResponse(
            text=Json_unpar_out.content
        )
    )
    knows_json = json_extract(expect_key="SupplementKnowledge", json_dict=Json_pure_out)
    knows_json_msg = Msg(name="SupplementKnowledge", content=knows_json)
    topic_json = json_extract(expect_key="Topic", json_dict=Json_pure_out)
    topic_json_msg = Msg(name="Topic", content=topic_json)
    scene_json = json_extract(expect_key="Scenarios", json_dict=Json_pure_out)
    scene_json_msg = Msg(name="Scenarios", content=scene_json)
    eval_json = json_extract(expect_key="EvaluationMetrics", json_dict=Json_pure_out)
    eval_json_msg = Msg(name="EvaluationMetrics", content=eval_json)

    # Step4: Refine the knowledge points.
    Refined_Know = KnowledgeRefiner(knows_json_msg)

    # Step5: Check if eg_in_out is empty.
    if Eg_input.content != "":
        msg_ini_prompt = msg_concat(Example_Input=Eg_input, Example_Output=Eg_output, User_Requirements=Query,
                                    Topic=topic_json_msg, Refined_Knowledges=Refined_Know)
        init_prompt_out = IniPromptGenerator_eg(msg_ini_prompt)
    else:
        msg_ini_prompt = msg_concat(User_Requirements=Query, Topic=topic_json_msg, Refined_Knowledges=Refined_Know)
        init_prompt_out = IniPromptGenerator(msg_ini_prompt)

    init_prompt_pure_out = json_parser.parse(
        ModelResponse(
            text=init_prompt_out.content
        )
    )
    initial_prompt = json_extract(expect_key="Initial Prompt", json_dict=init_prompt_pure_out)
    initial_prompt_msg = Msg(name="Initial Prompt", content=initial_prompt)

    # Step6: Generate evaluation data.
    msg_valid_gen = msg_concat(User_Requirements=Query, Topic=topic_json_msg, Scenarios=scene_json_msg,
                               Refined_Knowledges=Refined_Know)
    validation_out = ValidationGenerator(msg_valid_gen)
    validation_pure_out = json_parser.parse(
        ModelResponse(
            text=validation_out.content
        )
    )
    validation_json = json_extract(expect_key="Validation", json_dict=validation_pure_out)
    validation_json_msg = Msg(name="Validation", content=validation_json)

    # Step7: Generate the corresponding answers for the validation data.
    msg_QA = msg_concat(Validation_Data=validation_json_msg, Initial_Prompt=initial_prompt_msg)
    QA_out = QAGenerator(msg_QA)
    QA_pure_out = json_parser.parse(
        ModelResponse(
            text=QA_out.content
        )
    )
    QA_json = json_extract(expect_key="QA Pairs", json_dict=QA_pure_out)
    QA_json_msg = Msg(name="QA Pairs", content=QA_json)

    # Step8: Generate evaluation output.
    msg_eval = msg_concat(QA_Pairs=QA_json_msg, Evaluation_Metrics=eval_json_msg,
                          Initial_Prompt=initial_prompt_msg)
    suggestion_out = Evaluator(msg_eval)
    suggestion_pure_out = json_parser.parse(
        ModelResponse(
            text=suggestion_out.content
        )
    )
    suggestion_json = json_extract(expect_key="Suggestions", json_dict=suggestion_pure_out)
    suggestion_json_msg = Msg(name="Suggestions", content=suggestion_json)

    # Step9: Revise the Initial Prompt and output the Final Prompt template.
    msg_final_prompt = msg_concat(Suggestions=suggestion_json_msg, Initial_Prompt=initial_prompt_msg)
    final_prompt_out = PromptRevisor(msg_final_prompt)
    # final_prompt_pure_out = json_parser.parse(
    #     ModelResponse(
    #         text=final_prompt_out.content
    #     )
    # )
    # final_prompt_json = json_extract(expect_key="Final Prompt", json_dict=final_prompt_pure_out)
    return final_prompt_out


@app.route('/generate_prompt', methods=['POST'])
def handle_generate_prompt():
    data = request.json
    query = data.get('query', "")
    eg_in = data.get('eg_in', "")
    eg_out = data.get('eg_out', "")

    result = generate_prompt(query, eg_in, eg_out)
    return result.content


if __name__ == '__main__':
    app.run(host='192.168.31.131', port=5000)
