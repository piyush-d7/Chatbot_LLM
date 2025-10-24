# from openai import OpenAI

# ##### API 配置 #####
# openai_api_key = "NjgzNmZkZjU0ZTEwZDU4ZWQ1MDkzZWE4YWVkMzhiMmI2YzhkZmI0MA=="
# openai_api_base = "http://5660157692410416.us-east-1.pai-eas.aliyuncs.com/api/predict/chatbot_llm/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# ) 

# models = client.models.list()
# model = models.data[0].id
# print(model)


# def main():

#     stream = True

#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "You are an expert customer service assistant for Thermernergy. --- YOUR DIRECTIVES --- 1. **Analyze User Intent First:** - If the user shows clear buying intent (asking for a quote, price, to book a service, or wanting to contact the team), your ONLY response MUST be the special command: __SHOW_LEAD_FORM__ - The ONLY EXCEPTION to this rule is if the user's question explicitly asks for a 'phone number', 'email address', or 'physical address'. In that case, you may provide it from the retrieved context. 2. **Handle Conversational Replies:** - If the user's message is a simple conversational reply (e.g., 'thanks', 'okay', 'that's good'), respond naturally and politely without searching for information. 3. **Answer Informational Questions:** - If the user is asking a specific question, answer it based *only* on the 'Retrieved Context'. - If the answer is not in the context, politely say 'I'm sorry, I couldn't find specific information on that. Would you like me to connect you with a specialist?' --- --- Retrieved Context --- {heat pump, oil gas, air conditioner} ------------------------- --- Chat History --- {formatted_history} -------------------- New User Question: '{Tell me service you provide}' Your Answer:",
#                     }
#                 ],
#             }
#         ],
#         model=model,
#         max_completion_tokens=2048,
#         stream=stream,
#     )

#     if stream:
#         for chunk in chat_completion:
#             print(chunk.choices[0].delta.content, end="")
#     else:
#         result = chat_completion.choices[0].message.content
#         print(result)


# if __name__ == "__main__":
#     main()

from openai import OpenAI 
 
##### API 配置 ##### 
openai_api_key = "NjgzNmZkZjU0ZTEwZDU4ZWQ1MDkzZWE4YWVkMzhiMmI2YzhkZmI0MA==" 
openai_api_base = "http://5660157692410416.us-east-1.pai-eas.aliyuncs.com/api/predict/chatbot_llm/v1" 
 
client = OpenAI( 
    api_key=openai_api_key, 
    base_url=openai_api_base, 
)  
 
models = client.models.list() 
model = models.data[0].id 
print(model) 
 
 
def main(): 
    stream = True 
    
    # Build the complete prompt as a single string
    prompt = """You are an expert customer service assistant for Thermernergy. 

--- YOUR DIRECTIVES --- 
1. **Analyze User Intent First:** 
   - If the user shows clear buying intent (asking for a quote, price, to book a service, or wanting to contact the team), your ONLY response MUST be the special command: __SHOW_LEAD_FORM__ 
   - The ONLY EXCEPTION to this rule is if the user's question explicitly asks for a 'phone number', 'email address', or 'physical address'. In that case, you may provide it from the retrieved context. 

2. **Handle Conversational Replies:** 
   - If the user's message is a simple conversational reply (e.g., 'thanks', 'okay', 'that's good'), respond naturally and politely without searching for information. 

3. **Answer Informational Questions:** 
   - If the user is asking a specific question, answer it based *only* on the 'Retrieved Context'. 
   - If the answer is not in the context, politely say 'I'm sorry, I couldn't find specific information on that. Would you like me to connect you with a specialist?' 

--- Retrieved Context --- 
{heat pump, oil gas, air conditioner} 
------------------------- 

--- Chat History --- 
{formatted_history} 
-------------------- 

New User Question: 'Tell me service you provide' 

Your Answer:"""
 
    completion = client.completions.create( 
        model=model, 
        prompt=prompt,
        max_tokens=2048,
        stream=stream, 
    ) 
 
    if stream: 
        for chunk in completion: 
            print(chunk.choices[0].text, end="") 
    else: 
        result = completion.choices[0].text 
        print(result) 
 
 
if __name__ == "__main__": 
    main()