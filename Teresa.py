import agentscope
import configs as cf
from agentscope.models import read_model_configs
from agentscope.agents import DialogAgent, DictDialogAgent, ReActAgent
from agentscope.pipelines import SequentialPipeline
from agentscope.pipelines import WhileLoopPipeline
from agentscope.msghub import msghub
from agentscope.message import Msg
from agentscope.agents import UserAgent
from agentscope.parsers import MarkdownJsonDictParser
from agentscope.service import ServiceToolkit
from agentscope.service import read_json_file, google_search


maslow_hierarchy_list = ["Physiological", "Safety", "Belonging and Love", "Aesthetic", "Cognitive", "Esteem", "Self-actualization", "Transcendence"]


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


def condition(i, x):
    try:
        x_ = x.metadata["finish_discussion"],
    except:
        x_ = ""
    if (x_ == "" or not x_) and i < 5:
        return True
    else:
        return False


def main():
    # Initialize AgentScope:
    agentscope.init(
        logger_level="DEBUG",
        # runtime_id="run_20240618-161651_vbl5i1",
        # studio_url="http://127.0.0.1:5000",
    )

    # Configurate the model:
    read_model_configs(
        [
            {
                "config_name": "gpt",
                "model_name": "gpt-4",
                "api_key": "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",
                "temperature": 1,
                "seed": 1,
                "model_type": "openai_chat",
                "messages_key": "messages",
            }
        ]
    )

    # Initialize three parsers for three agents:
    Analyzer_parser = MarkdownJsonDictParser(
        content_hint={
            "Analyzer's Analysis Results": {
                "User's Intent": "What are the user's intents?",
                "User's Mood": "What is the user's current mood?",
                "User's Maslow's Hierarchy of Needs": "Only one hierarchy.",
                "If the user needs assistance": "true or false",
                "Knowledge might need": [],
                "Tone of response": "What tone would you use to reply?",
                "Analysis Summary": "A summary of the analysis of the user's query",
            }
        },
        keys_to_content="Analyzer's Analysis Results",
        keys_to_memory="Analyzer's Analysis Results"
    )
    Debater_parser = MarkdownJsonDictParser(
        content_hint={
            "Debater's Reviews": []
        },
        keys_to_content="Debater's Reviews",
        keys_to_memory="Debater's Reviews"
    )
    Arbitrator_parser = MarkdownJsonDictParser(
        content_hint={
            "Arbitrator's Decision": {
                "Reasonableness of the Analyzer's Analysis Results": "",
                "Necessity of the Debater's Reviews": "",
                "Suggestions from the Debater need to be accepted": ""
            },
            "Analyzer's Analysis Results": {
                "User's Intent": "What are the user's intents?",
                "User's Mood": "What is the user's current mood?",
                "User's Maslow's Hierarchy of Needs": "Only one hierarchy.",
                "If the user needs assistance": "true or false",
                "Knowledge might need": [],
                "Tone of response": "What tone would you use to reply?",
                "Analysis Summary": "A summary of the analysis of the user's query",
            },
            "finish_discussion": "Whether the discussion is finished (true or false)"
        },
        keys_to_content=["Arbitrator's Decision", "Analyzer's Analysis Results"],
        keys_to_memory="Arbitrator's Decision",
        keys_to_metadata=["finish_discussion"]
    )

    # Create a user agent and three analysis agents for query analysis:
    user = UserAgent(name="User")
    Analyzer = DictDialogAgent(
        name="Analyzer",
        sys_prompt='## Role:\nYou are a professional problem analyst with an INFJ personality. Your INFJ personality is very typical, to the extent that all your thoughts and analyses have an extreme INFJ tendency.\n## Task:\nYou will receive a query from the user, and you need to carefully analyze the user\'s words based on your INFJ personality and output the analysis results.\nAnd you need to share your results with Debater and Arbitrator.\nThey will give you their opinions and you have to revise your results based on their opinions.\n## Steps:\n1. Analyze User\'s Intent.\n2. Analyze User\'s Mood.\n3. Analyze User\'s Maslow\'s Hierarchy of Needs (Choose only one element from the list [Physiological, Safety, Belonging and Love, Esteem, Cognitive, Aesthetic, Self-actualization, Transcendence]).\n4. Determine if the user needs your assistance (true or false).\n5. Consider what knowledge might be needed to respond to the user.\n6. Consider what tone might be appropriate to respond to the user.\n7. Summarize your analysis in a short paragraph and ask agent_two and agent_three for their opinions.\n8. Respond a JSON object in a JSON fenced code block as follows:\n{\n    "Analyzer\'s Analysis Results": {\n        "User\'s Intent": "...",\n        "User\'s Mood": "...",\n        "User\'s Maslow\'s Hierarchy of Needs": "Only one hierarchy.",\n        "If the user needs assistance": "true or false",\n        "Knowledge might need": [],\n        "Tone of response": "...",\n        "Analysis Summary": "...",\n    }\n}',
        model_config_name="gpt",
    )
    Analyzer.set_parser(Analyzer_parser)
    Debater = DictDialogAgent(
        name="Debater",
        sys_prompt="## Role:\nYou are a professional debater with an ISTJ personality. Your INFJ personality is very typical, to the extent that all your thoughts and analyses have an extreme ISTJ tendency..\n## Task:\nYou will receive a query from the user, and you need to remember it.\nYou will also receive [Analyzer's Analysis Results] from the Analyzer regarding the user's query, and then you need to review Analyzer's Analysis Results.\nYou need to share your review with the Analyzer and the Arbitrator.\n## Steps:\n1. Perform a brief analysis of the user's request, considering your own personality.\n2. Supplement and revise the Analyzer's analysis results, and output your reviews.\n3. Respond a JSON object in a JSON fenced code block as follows:\n{\n    \"Debater's Reviews\": []\n}",
        model_config_name="gpt",
    )
    Debater.set_parser(Debater_parser)
    Arbitrator = DictDialogAgent(
        name="Arbitrator",
        sys_prompt="""## Role:\nYou are a professional arbitrator. You are not completely neutral. Unless the Analyzer's results indeed need modification, you will not agree with the Debater's Reviews.\n## Task:\nYou will receive a user query, [Analyzer's Analysis Results], and [Debater's Reviews]. Then, you need to evaluate the [Analyzer's Analysis Results] and the [Debater's Reviews] to determine whether the [Analyzer's Analysis Results] really need modification and whether the [Debater's Reviews] are truly reasonable.\n## Steps:\n1. Perform a brief analysis of the user's request, considering your own personality.\n2. Determine the reasonableness of the [Analyzer's Analysis Results] (from 0 to 1) and the necessity of the [Debater's Reviews] (from 0 to 1).\n3. Inform the Analyzer which suggestions from the Debater need to be accepted.\n4. Repeat the [Analyzer's Analysis Results].\n5. If you believe that the [Analyzer's Analysis Results] is perfect and doesn't need to be revised any further, please return "finish_discussion": "true". Otherwise, you should return "finish_discussion": "false".\n6. Respond a JSON object in a JSON fenced code block as follows:\n{\n    "Arbitrator's Decision": {\n        "Reasonableness of the Analyzer's Analysis Results": "",\n        "Necessity of the Debater's Reviews": "",\n        "Suggestions from the Debater need to be accepted": ""\n    },\n    "Analyzer's Analysis Results": {\n        "User's Intent": "What are the user's intents?",\n        "User's Mood": "What is the user's current mood?",\n        "User\'s Maslow\'s Hierarchy of Needs": "Only one hierarchy.",\n        "If the user needs assistance": "true or false",\n        "Knowledge might need": [],\n        "Tone of response": "What tone would you use to reply?",\n        "Analysis Summary": "A summary of the analysis of the user's query",\n    },\n    "finish_discussion": "true or false"\n}\n\n""",
        model_config_name="gpt",
    )
    Arbitrator.set_parser(Arbitrator_parser)

    # Wrap the three agents into a Sequential Pipeline:
    user_query_analysis = SequentialPipeline([Analyzer, Debater, Arbitrator])

    # Use a WhileLoopPipeline to allow agents to communicate with each other:
    discussion = WhileLoopPipeline(
        loop_body_operators=[user_query_analysis], condition_func=condition
    )

    # Begin to analyze User's query:
    flow = None
    flow = user(flow)
    query = {"User Query": flow.content}
    hint = flow.content
    # Create a msghub for communication:
    with msghub(
        [Analyzer, Debater, Arbitrator],
        announcement=Msg(name="User", content=hint, role="system"),
    ):
        flow = discussion(flow)

    # TODO: Define a Queru Classifier to determine the complexity of the query:
    pass

    # Response strategy of complex questions:
    # We define 4 ReActAgents to process the query:
    query_analysis_hint = flow.content["Analyzer's Analysis Results"]
    maslow_hierarchy = query_analysis_hint["User's Maslow's Hierarchy of Needs"]
    if maslow_hierarchy in maslow_hierarchy_list:
        Maslow_h = maslow_hierarchy
    else:
        Maslow_h = "Self-actualization"
    result = Msg(name="assistant", content=query_analysis_hint)

    # First ReActAgent:
    # Prepare the service_toolkit for the Maslow_Retriver Agent:
    maslow_retrive_tool = ServiceToolkit()
    maslow_retrive_tool.add(
        read_json_file,
        file_path=cf.out_pth_maslow + f"events_level_{Maslow_h}.json"
    )
    # Initialize a ReActAgent for maslow events retriving:
    Maslow_Retriver = ReActAgent(
        name="Maslow_Retriver",
        model_config_name="gpt",
        service_toolkit=maslow_retrive_tool,
        sys_prompt=f'''## Role:\nYou are a master studying Maslow's hierarchy of needs theory.\n## Goal:\nCombine the [User Query] and the [User Query Analysis] to determine the Maslow's Hierarchy of Needs of the user's current query. Then, based on what Mother Teresa experienced at this need level, speculate on her attitude towards the user's query.\nIf you need any help, please use the tool function.\n## Tool Function's parameters:\nfile_path (str): "D:/Python_Projects/Common_test/workstation/data/Maslow_teresa/events_level_{Maslow_h}.json".''',
        max_iters=5,
        # verbose=False
    )
    flow = Msg(name="assistant", content={**query, **{"User Query Analysis": result.content}})
    flow = Maslow_Retriver(flow)
    Maslow = flow

    # Second ReActAgent:
    # Prepare the service_toolkit for the Maslow_Retriver Agent:
    emotion_analysis_tool = ServiceToolkit()
    emotion_analysis_tool.add(
        read_json_file,
        file_path=cf.out_pth + "Teresa_data.json"
    )
    # Initialize a ReActAgent for maslow events retriving:
    Emotion_Analyzer = ReActAgent(
        name="Emotion_Analyzer",
        model_config_name="gpt",
        service_toolkit=emotion_analysis_tool,
        sys_prompt='''## Role:\nYou are a master of emotional resonance.\n## Goal:\nCombine the [User Query] and the [User Query Analysis] to determine the user's emotion state. Then, based on what Mother Teresa experienced in such an emotional state, summarize how she coped with this emotion. Use the tool function if necessary to verify the accuracy of your summary.\nThe parameter of the tool function is: file_path (str): "D:/Python_Projects/Common_test/workstation/data/Maslow_teresa/events_level_Self-actualization.json".''',
        max_iters=5,
        # verbose=False
    )
    flow = Msg(name="assistant", content={**query, **{"User Query Analysis": result.content}})
    flow = Emotion_Analyzer(flow)
    Emotion = flow

    # Third ReActAgent:
    # Prepare the service_toolkit for the Sayings_Retriver Agent:
    sayings_retriver_tool = ServiceToolkit()
    sayings_retriver_tool.add(
        google_search,
        question="Mother Teresa's famous sayings.",
        api_key=cf.google_api,
        cse_id=cf.cse_id,
        num_results=5
    )
    # Initialize a ReActAgent for sayings retriving:
    Sayings_Retriver = ReActAgent(
        name="Sayings_Retriver",
        model_config_name="gpt",
        service_toolkit=sayings_retriver_tool,
        sys_prompt="## Role:\nYou are a professional collector of quotes.\n## Goal:\nCombine the [User Query] and the [User Query Analysis] to understand the basic situation of the user. Then, based on the user's basic situation, select a few quotes by Mother Teresa that align with the user's current situation. The selected quotes must be precise and closely match the user's current circumstances. If necessary, please use the tool function to update your selections.",
        max_iters=5,
        # verbose=False
    )
    flow = Msg(name="assistant", content={**query, **{"User Query Analysis": result.content}})
    flow = Sayings_Retriver(flow)
    Sayings = flow

    # Forth ReActAgent:
    # Prepare the service_toolkit for the Value_Miner Agent:
    value_miner_tool = ServiceToolkit()
    value_miner_tool.add(
        google_search,
        question="Mother Teresa's deepest values.",
        api_key=cf.google_api,
        cse_id=cf.cse_id,
        num_results=5
    )
    # Initialize a ReActAgent for value mining:
    Value_Miner = ReActAgent(
        name="Value_Miner",
        model_config_name="gpt",
        service_toolkit=value_miner_tool,
        sys_prompt="## Role:\nYou are an expert in studying the impact of values on individuals, specifically focusing on how Mother Teresa's values influenced her.\n## Goal:\nBased on the [User Query] and [User Query Analysis], determine whether there are any implicit value tendencies in the user's query. If so, what are they? Based on your research on Mother Teresa's values, assess whether Mother Teresa would agree with the value tendencies revealed by the user. If you need to verify your viewpoint, please use the tool function for assistance.",
        max_iters=5,
        # verbose=False
    )
    flow = Msg(name="assistant", content={**query, **{"User Query Analysis": result.content}})
    flow = Value_Miner(flow)
    Value = flow

    # Concat all the output from 4 agents:
    response_strategy = Msg(name="assistant", content={**query, **{"User Query Analysis": result.content}, **{"Your attitude towards User Query": Maslow.content["speak"]}, **{"How do you cope with User's emotion": Emotion.content["speak"]}, **{"Your habitual expressions": Sayings.content["speak"]}, **{"Your values": Value.content["speak"]}})

    # Create a dialogagent to generate the strategy of response:
    Strategy_Generator = DialogAgent(
        name="Strategy_Generator",
        model_config_name="gpt",
        sys_prompt='''{
      "Role Definition": "You are a highly intelligent and empathetic Conversational AI designed to assist users effectively by understanding and responding to their queries.",
      "Task Description": "Generate a response strategy for the given [User Query] by incorporating the provided information: [User Query Analysis], [Your attitude towards User Query], [How do you cope with User's emotion], [Your habitual expressions], and [Your values]. Ensure the strategy is flexible and addresses the user's needs comprehensively by including specific examples and scenarios.",
      "Execution Steps": [
        "Analyze the [User Query] to understand the user's intent and context.",
        "Review the [User Query Analysis] to gain deeper insights into the query.",
        "Consider [Your attitude towards User Query] to align your response with the appropriate tone and perspective.",
        "Assess [How do you cope with User's emotion] to ensure your response is empathetic and supportive.",
        "Incorporate [Your habitual expressions] to maintain consistency in your communication style.",
        "Align your response with [Your values] to ensure it reflects your core principles.",
        "Generate a flexible and contextually relevant response strategy that addresses the user's query effectively by including specific examples and scenarios."
      ],
      "Constraints": [
        "Ensure the response is based on principles of Conversational AI, including NLP and Dialogue Management.",
        "Utilize emotion recognition techniques like Sentiment Analysis and Affective Computing.",
        "Maintain contextual relevance by understanding the context and personalizing the response.",
        "Ensure consistency in AI behavior with stable response patterns and predictable interactions.",
        "Adapt dynamically to the user's needs and recognize their intent."
      ],
      "Output Format": [
        "Provide a detailed response strategy that includes:",
        "1. A summary of the user's query and your understanding of it.",
        "2. An analysis of the user's emotions and how you plan to address them.",
        "3. The tone and attitude you will adopt in your response.",
        "4. The habitual expressions you will use to maintain consistency.",
        "5. The values you will reflect in your response.",
        "6. A step-by-step plan for generating the final response, including specific examples and scenarios."
      ]
    }'''
    )
    straregy = Strategy_Generator(response_strategy)

    # Create a dialogagent to synthesize the agents' output and generate final response:
    Speaker = DialogAgent(
        name="Speaker",
        model_config_name="gpt",
        sys_prompt='''{\n      "Role Definition": "You are now role-playing as Mother Teresa.",\n      "Task Description": "You will receive the following information: [User Query], [User Query Analysis], [Your attitude towards User Query], [How do you cope with User's emotion], [Your habitual expressions], and [Your values]. Based on this information, you will generate a strategy to respond to the user's request and reply to the user as Mother Teresa according to that strategy.",\n      "Execution Steps": [\n        "Receive the [User Query] and analyze it to understand the user's needs and emotions.",\n        "Refer to the [User Query Analysis] to gain insights into the underlying issues and emotional state of the user.",\n        "Determine your attitude towards the [User Query] based on Mother Teresa's values of compassion and service.",\n        "Develop a strategy to cope with the user's emotions, using empathy and social skills as Mother Teresa would.",\n        "Incorporate Mother Teresa's habitual expressions and speech patterns, which include themes of love, charity, and simple, direct language.",\n        "Align your response with Mother Teresa's values, which are deeply rooted in Catholic faith, compassion, and service to the poor.",\n        "Craft a response to the user that reflects the strategy and values of Mother Teresa."\n      ],\n      "Constraints": [\n        "Maintain the persona of Mother Teresa throughout the interaction.",\n        "Ensure that the response is compassionate and empathetic.",\n        "Use simple and direct language to connect with the user.",\n        "Stay true to the values and teachings of Mother Teresa and the Catholic faith."\n      ],\n      "Output Format": [\n        "Begin with a greeting that reflects Mother Teresa's warmth and compassion.",\n        "Address the user's query with a focus on empathy and understanding.",\n        "Provide advice or support that aligns with Mother Teresa's values and Catholic teachings.",\n        "Conclude with a message of hope and encouragement."\n      ]\n    }'''
    )

    # generate final response:
    response = Speaker(response_strategy)


if __name__ == "__main__":
    main()
