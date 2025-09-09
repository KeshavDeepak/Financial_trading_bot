from backend.parser_config import *

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ollama import chat

#* constants
MODEL = 'gemma3:4b' 

#* fastapi app 
app = FastAPI()

#* ensure front-end is allowed to talk to back-end
app.add_middleware(
    CORSMiddleware,
    # allow_origins=['http://localhost:5173'],  #* frontend URL
    allow_origins=['*'],  #* frontend URL
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

#* endpoint that parses an user prompt into a command and sends it back
@app.post('/parse-command')
async def parse_command(request: Request):
    data = await request.json()

    user_prompt = data.get('user_prompt', '')

    #* add the system instructions and the user prompt to a list
    messages = []
    messages.append(SETUP_MODEL_FOR_USER_PROMPT)
    messages.append({'role' : 'user', 'content' : user_prompt})

    #* call the ollama model
    response = chat(model=MODEL, messages=messages)
    command = response.message.content.strip()
    
    #* split the command up into its components
    command_components = normalize_command(command)
    
    #* filter through commands, utilize the appropriate function, and generate a response
    match command_components[0]:
        case 'show':
            # resonse is an img64
            response = show(command_components[1], command_components[2], command_components[3]) 
            
        case 'suggest':
            '''
            get the response from the agent and also a system_prompt explaining to our ollama model how to reframe the answer
            into something human-readable
            '''
            answer, system_prompt = suggest(command_components[1]) 
            
            #* create a temporary conversation log with the system_prompt and the answer for our ollama model
            temp = [
                {'role' : 'system', 'content' : system_prompt},
                {'role' : 'user', 'content' : answer},
            ]
            
            #* call the ollama model and get a human-readable response back
            response = chat(model=MODEL, messages=temp)
            response = response.message.content.strip()
            
        case 'error':
            print('error')
        
    return {
        'command' : command_components[0],
        'answer' : response
    }


'''
The various commands the client can parse are :
OLD -- buy [asset] <shares> 
OLD -- sell [asset] <shares>
above two has been refactored into one function --> suggest [ticker]

predict [asset] [time_period]
show [asset] [time_period]
compare [asset_1] vs [asset_2]
explain [concept]
'''
