import { useState } from "react";

export default function Chatbox({ handleUserInput }) {
    const [input, setInput] = useState("")

    const handleSubmit = (e) => {
        e.preventDefault(); // ensures the page does not refresh

        if (!input.trim()) return; // ignore empty messages

        handleUserInput(input); // call parent component's function to handle user input

        setInput(""); // clear the input
    }

    return (
        <form id="chatbox-form" onSubmit={handleSubmit} >
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask something..."
            />
        </form>
    )
}