export default function Dialogue({ messages }) {
    if (messages.length == 0) {
        return (
            <div id='dialogue'>
                {
                    messages.map((msg) => (
                        <div key={msg.id} className={`message ${msg.speaker === "user" ? "user" : "bot"}`}>
                            {msg.text}
                        </div>
                        ))
                }
            </div>
        )
    }
}