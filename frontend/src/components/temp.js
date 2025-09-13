import React, { useState, useRef } from 'react';
import { Send, Square, Paperclip, FileText, X } from 'lucide-react';

// Your existing component with the modified structure
const ChatForm = ({ 
  hasUploadedDocuments, 
  uploadedFileName, 
  deleteRagSession, 
  handleSendMessage, 
  handleFileAttach, 
  inputText, 
  setInputText, 
  isTyping, 
  handleStopGeneration, 
  fileInputRef, 
  handleFileSelect, 
  models, 
  currentModel 
}) => {
  return (
    <form onSubmit={handleSendMessage} className="chat-input-form">
      <div className="input-container">
        <button
          type="button"
          onClick={handleFileAttach}
          className="attach-button"
          title="Attach file for RAG processing (PDF, TXT, DOCX)"
          disabled={isTyping}
        >
          <Paperclip size={20} />
        </button>
        
        <div className="textarea-container">
          {hasUploadedDocuments && uploadedFileName && (
            <div className="file-indicator-inline">
              <div className={`file-item ${uploadedFileName.split('.').pop().toLowerCase()}`}>
                <FileText size={16} className="file-icon" />
                <span className="file-name">{uploadedFileName}</span>
                <button
                  onClick={deleteRagSession}
                  className="file-delete-button"
                  title="Remove uploaded file"
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          )}
          
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={`Message Varuna (${models[currentModel].name})${hasUploadedDocuments ? ' - RAG enabled' : ''}`}
            className="chat-input"
            disabled={isTyping}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault(); // Prevent new line
                if (!isTyping && inputText.trim()) {
                  handleSendMessage(e); // Call your submit function
                }
              }
            }}
            rows={1}
          />
        </div>
        
        <button
          type="button"
          className={`send-button ${isTyping ? 'stop-button' : ''}`}
          disabled={(!inputText.trim() && !isTyping)}
          title={isTyping ? "Stop generation" : "Send message"}
          onClick={isTyping ? handleStopGeneration : handleSendMessage}
        >
          {isTyping ? <Square size={20} /> : <Send size={20} />}
        </button>
      </div>
      
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        style={{ display: 'none' }}
        accept=".pdf,.txt,.docx,.doc,.png,.jpeg,.jpg"
      />
    </form>
  );
};