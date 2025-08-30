export default function Dialogue({ messages }) {
    if (messages.length != 0) {
        const chat_bubbles = [];

        for (let msg of messages) {
            if (msg.role == "user") {
                chat_bubbles.push(
                    <div key={msg.id} className={`message user`}>
                        {msg.content}
                    </div>
                )
            }
            else if (msg.role == "assistant") {
                if (msg.command == "show") {
                    chat_bubbles.push(
                        <img key={msg.id} src={`data:image/png;base64,${msg.content}`} />
                    )
                }
                else {
                    chat_bubbles.push(
                        <div key={msg.id} className={`message assistant`}>
                            {msg.content}
                        </div>
                    )
                }
            }
        }

        return (
            <div id="dialogue">
                {chat_bubbles}
            </div>
        );
    }
}