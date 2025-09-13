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
            #* response is an img64
            response = show(command_components[1], command_components[2], command_components[3]) 
            
        case 'suggest':
            '''
            get the response from the agent and also a system_prompt explaining to our ollama model how to reframe the answer
            into something human-readable
            '''
            answer, system_prompt = suggest(command_components[1]) 
            
            #* temporary conversation log with a system_prompt and the answer for our ollama model
            temp = [
                {'role' : 'system', 'content' : system_prompt},
                {'role' : 'user', 'content' : answer},
            ]
            
            #* call the ollama model and get a human-readable response back
            response = chat(model=MODEL, messages=temp)
            response = response.message.content.strip()
        
        case 'explain':
            #* temporary conversation log with a system_prompt to generate an explanation of the input concept
            temp = [
                {'role' : 'system',
                 'content' : f'''
                    Give a brief explanation of the concept {command_components[1]}.
                    The concept will always be related to the domain of finance and trading.
                    Keep your explanation to 100 words and do not use markup in your response.
                 '''}
            ]
            
            #* call the ollama model and get an explanation back
            response = chat(model=MODEL, messages=temp)
            response = response.message.content.strip()
        
        case 'backtest':            
            response = get_portfolio_plot(command_components[1])
            
        case 'error':
            print('error')
        
    return {
        'command' : command_components[0],
        'answer' : response
    }


'''
AI powered commands --
    suggest [ticker]
    backtest [ticker]
    
    compare [ticker1] vs [ticker2] ?

Visualization commands --
    show [asset] [time_period]
    explain [concept]


Utility and info commands --
    help ?
'''
