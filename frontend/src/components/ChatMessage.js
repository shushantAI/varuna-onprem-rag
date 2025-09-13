import React from 'react';
import ReactMarkdown from 'react-markdown';

const ChatMessage = ({ message }) => {
  const isUser = message.sender === 'user';
  
  return (
    <div className={`message ${isUser ? 'user-message' : 'varuna-message'}`}>
      <div className="message-content">
        <div className={`avatar ${isUser ? 'user-avatar' : 'varuna-avatar'}`}>
          {isUser ? 'U' : 'V'}
        </div>
        <div className="message-bubble">
          <div className="preserve-whitespace"><ReactMarkdown children={message.text} /></div>
          <span className="timestamp">
            {message.timestamp.toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;