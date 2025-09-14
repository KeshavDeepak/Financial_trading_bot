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
                if (msg.command == "show" || msg.command == "backtest") {
                    chat_bubbles.push(
                        <img 
                            key={msg.id}
                            className="chat-image" 
                            src={`data:image/png;base64,${msg.content}`} 
                        />
                    )
                }
                else if (msg.command == "help") {
                    chat_bubbles.push(
                        <div key={msg.id} className={`message assistant`}>
                            <p><b>The available commands are:</b></p>
                            <ul>
                                <li><code>suggest [ticker]</code></li>
                                <li><code>show [ticker] [start_time] [end_time]</code></li>
                                <li><code>explain [concept]</code></li>
                                <li><code>backtest [ticker]</code></li>
                                <li><code>help</code></li>
                            </ul>

                            <p><b>Functionality:</b></p>
                            <ul>
                                <li><code>suggest</code>: guidance on buy/sell</li>
                                <li><code>show</code>: see past stock data</li>
                                <li><code>explain</code>: explanation of a concept</li>
                                <li><code>backtest</code>: run backtest on ticker</li>
                                <li><code>help</code>: list available commands</li>
                            </ul>
                        </div>
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