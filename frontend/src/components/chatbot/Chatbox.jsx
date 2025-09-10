import { useEffect, useRef, useState } from "react";

export default function Chatbox({ handleUserInput }) {
    //* the live user input 
    const [input, setInput] = useState("")

    //* holds a reference textarea DOM element in textAreaRef.current
    const textAreaRef = useRef(null);

    //* when the user submits their input, the parent component's handleUserInput function is called
    const handleSubmit = (e) => {
        e.preventDefault(); //* ensures the page does not refresh

        if (!input.trim()) return; //* ignore empty messages

        handleUserInput(input); //* call parent component's function to handle user input

        setInput(""); //* clear the input
    }

    //* auto resize text-area when input state changes (user has typed something)
    useEffect(() => {
        if (textAreaRef.current) {
            textAreaRef.current.style.height = "auto"; //* reset height (for shrinking)
            textAreaRef.current.style.height = textAreaRef.current.scrollHeight + "px"; //* increase height (for expanding)
        }
    }, [input]);


    //* a text area for the user to type their queries in
    return (
        <form id="chatbox-form" onSubmit={handleSubmit} >
            <textarea 
                id="chatbox-textarea"
                ref={textAreaRef}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault(); // prevent newline
                        handleSubmit(e);     // call handleSubmit
                    }
                }}
                placeholder="Ask something..."
            />
        </form>
    )
}