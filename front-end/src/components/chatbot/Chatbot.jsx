import { useState } from 'react';
import './Chatbot.css'

import Dialogue from './Dialogue';
import Chatbox from './Chatbox';

export default function Chatbot() {
    //* states
    const [messages, setMessages] = useState([]);

    //* handle user input 
    const handleUserInput = (user_input) => {
        let messages_length = messages.length

        //* draft a new message with the text as that in user_input
        const newMsg = {
            speaker : 'user',
            text : user_input,
            id : messages_length + 1
        }

        //* update messages state to include this new message
        setMessages((prev_list) => [...prev_list, newMsg])

        //* simulate bot reply
        setTimeout(() => {
            setMessages((prev_list) => [
                ...prev_list,
                { speaker : 'bot', 
                  text : 'hi I am bot and I responded to your message automatically',
                  id : messages.length + 2} // length of messages does not update synchronously 
            ])
        })
    }

    return (
        <>
            <Dialogue messages={messages}/>  
            <Chatbox handleUserInput={handleUserInput}/>
        </>
    )
}