from backend.parser_config import *

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ollama import chat

#* name of model
MODEL = 'gemma3:4b' 

#* fastapi app 
app = FastAPI()

#* ensure front-end is allowed to talk to back-end
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],  #* frontend URL
    allow_origins=["*"],  #* frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#* endpoint that parses an user prompt into a command and sends it back
@app.post('/parse-command')
async def parse_command(request: Request):
    data = await request.json()

    user_prompt = data.get('user_prompt', '')

    #* add the system instructions and the user prompt to a list
    messages = []
    messages.append(SETUP_MODEL_PROMPT)
    messages.append({'role' : 'user', 'content' : user_prompt})

    #* call the ollama model
    response = chat(model=MODEL, messages=messages)
    command = response.message.content.strip()
    
    #* split the command up into its components
    command_components = normalize_command(command)
    
    #* filter through commands and utilize the appropriate function
    match command_components[0]:
        case "show":
            img64 = show(command_components[1], command_components[2], command_components[3]) 
            
            return {
                "command" : command_components[0],
                "answer" : img64
            }
        case "buy":
            #!!! what to do now
            pass
        case "error":
            print('error')



