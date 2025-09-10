import { useState } from 'react';
import './Chatbot.css'; 

import Dialogue from './Dialogue';
import Chatbox from './Chatbox';

export default function Chatbot() {
    //* messages
    const [messages, setMessages] = useState([]);

    //* handle user input 
    const handleUserInput = (user_prompt) => {
        let messages_length = messages.length;

        //* draft a new message with the text as that in user_prompt
        const newMsg = {
            role : 'user',
            content : user_prompt,
            id : messages_length + 1,
        };

        //* update messages state to include this new message
        setMessages((prev_list) => [...prev_list, newMsg]);
        
        //* get response
        sendNewMessage(user_prompt).then(response => {

            //* add response to messages state
            let bot_response = {
                role : 'assistant',
                content : response.answer,
                id : messages_length + 2,
                command : response.command
            };
            
            setMessages((prev_list) => [...prev_list, bot_response]);
        })
    }

    return (
        <>
            <Dialogue messages={messages}/>  
            <Chatbox handleUserInput={handleUserInput}/>
        </>
    )
}

//* send user prompt to llm for parsing
const sendNewMessage = async (user_prompt) => {   
    var response = await fetch("http://127.0.0.1:8000/parse-command", {
        method : "POST",
        headers : { "Content-type" : "application/json"},
        body : JSON.stringify({
            user_prompt : user_prompt
        })
    });

    response = await response.json();

    return response
};