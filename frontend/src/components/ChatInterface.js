import React, { useState, useRef } from 'react';
import { Anchor,Send, Paperclip, LogOut, User, Moon, Sun, Trash2, Bot, X, FileText, Square, Turtle } from 'lucide-react';
import ChatMessage from './ChatMessage';

//http://127.0.0.1:8000
const BACKEND_PATH='http://localhost:8000'

const ChatInterface = ({ user, onLogout, darkMode, toggleDarkMode }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: `Hello ${user.name}! I'm Varuna, an AI assistant. How can I help you today?`,
      sender: 'varuna',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [currentModel, setCurrentModel] = useState('model1'); // Default model
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [hasUploadedDocuments, setHasUploadedDocuments] = useState(false);
  const [ragSessionId, setRagSessionId] = useState(null); // Add RAG session management
  const [uploadedFileName, setUploadedFileName] = useState(''); // Store uploaded file name
  const [abortController, setAbortController] = useState(null); // For cancelling requests
  const fileInputRef = useRef(null);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const textareaRef = useRef(null);

  const adjustTextareaHeight = (el) => {
    el.style.height = 'auto'; 
    el.style.height = `${el.scrollHeight}px`;
  };

  // Model configurations
  const models = {
    model1: {
      name: 'Standard AI',
      id:'llama3.1:8b',
      icon: <Bot size={16} />,
      description: 'Balanced performance model'
    },
  };


  // Create RAG session
  const createRagSession = async () => {
    try {
      const response = await fetch(`${BACKEND_PATH}/rag/create-session`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setRagSessionId(data.session_id);
        return data.session_id;
      }
    } catch (error) {
      console.error('Error creating RAG session:', error);
    }
    return null;
  };

  // Delete RAG session and uploaded file
  const deleteRagSession = async () => {
    if (!ragSessionId) return;
    
    try {
      await fetch(`${BACKEND_PATH}/rag/document/${ragSessionId}`, {
        method: 'DELETE'
      });
      setHasUploadedDocuments(false);
      setUploadedFileName('');
    } catch (error) {
      console.error('Error clearing RAG documents:', error);
    }
  };

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (isDropdownOpen && !event.target.closest('.model-dropdown')) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  // Create RAG session on component mount
  React.useEffect(() => {
    const initializeRagSession = async () => {
      const sessionId = await createRagSession();
      if (sessionId) {
        // Check if session has documents
        try {
          const response = await fetch(`${BACKEND_PATH}/rag/session/${sessionId}/status`);
          if (response.ok) {
            const data = await response.json();
            setHasUploadedDocuments(data.has_document && data.chunks_count > 0);
          }
        } catch (error) {
          console.error('Error checking RAG status:', error);
        }
      }
    };

    initializeRagSession();
  }, [user]);

  // Initial greeting message
  const getInitialMessage = () => ({
    id: 1,
    text: `Hello ${user?.name}! I'm V, an AI assistant. How can I help you today?`,
    sender: 'varuna',
    timestamp: new Date()
  });

  // Function to convert messages array to string format
  const messagesToString = (messagesArray) => {
    return messagesArray.map(msg => {
      const role = msg.sender === 'user' ? 'User' : 'V';
      return `${role}: ${msg.text}`;
    }).join('\n');
  };

  const handleModelSwitch = async (newModel) => {
    if (newModel === currentModel) return;

    setCurrentModel(newModel);
    
    // Add a system message to the chat indicating model switch
    const modelSwitchMessage = {
      id: Date.now(),
      text: `Switched to ${models[newModel].name}`,
      sender: 'system',
      timestamp: new Date(),
      isSystemMessage: true
    };

    setMessages(prev => [...prev, modelSwitchMessage]);

    try {
      // Notify backend about model switch
      const response = await fetch(BACKEND_PATH+'/switch-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelName: models[newModel].id
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Model switch response:', data);
      
    } catch (error) {
      console.error('Error switching model:', error);
      
      // Add error message if model switch fails
      const errorMessage = {
        id: Date.now() + 1,
        text: `Failed to switch to ${models[newModel].name}. Please try again.`,
        sender: 'system',
        timestamp: new Date(),
        isSystemMessage: true,
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleSendMessage = async (e) => {
    textareaRef.current.style.height = 'auto';
    e.preventDefault();
    if (!inputText.trim() || isTyping || isThinking) return; // Prevent sending while typing
    // Convert chat history to string
    const chatHistoryString = messagesToString(messages);
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);
    setIsThinking(true);
    let thinking=true;
    // Create abort controller for this request
    const controller = new AbortController();
    setAbortController(controller);

    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    // Update messages with the new user message
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    // Create initial AI message that will be updated with streaming content
    const aiMessageId = Date.now() + 1;
    const initialAiMessage = {
      id: aiMessageId,
      text: '',
      sender: 'varuna',
      timestamp: new Date(),
      isStreaming: true
    };

    let accumulatedText = '';
    try {
      const response = await fetch(`${BACKEND_PATH}/message/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          chatHistory: chatHistoryString,
          model: currentModel,
          ragStatus: !!(hasUploadedDocuments && ragSessionId),
          session_id : ragSessionId,
          user: user.name
        }),
        signal: controller.signal // Add abort signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done){
          console.log("done")
          break;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data.trim() === '') continue;

            try {
              const parsed = JSON.parse(data);
              
              if (parsed.error) {
                throw new Error(parsed.error);
              }
              
              if (parsed.token) {
                if (thinking===true){
                  //disable 3 dots when stream starts
                  thinking=false
                  setIsThinking(false);
                  // Add the initial AI message to the chat
                  setMessages(prev => [...prev, initialAiMessage]);
                }
                // Accumulate the token
                accumulatedText += parsed.token;
                
                // Update the AI message with the new accumulated text
                // eslint-disable-next-line no-loop-func
                setMessages(prev => prev.map(msg => 
                  msg.id === aiMessageId 
                    ? { ...msg, text: accumulatedText +'...' }
                    : msg
                ));
              }
              
              if (parsed.done) {
                // Mark streaming as complete
                // eslint-disable-next-line no-loop-func
                setMessages(prev => prev.map(msg => 
                  msg.id === aiMessageId 
                    ? { ...msg,text: accumulatedText, isStreaming: false }
                    : msg
                ));
                console.log("parser finished")
                return; // Exit the function
              }
            } catch (parseError) {
              console.error('Error parsing streaming data:', parseError);
              const errorMessage = {
                id: Date.now(),
                text: `❌ Sorry, I encountered an error while sending response to server. Please try again.`,
                sender: 'varuna',
                timestamp: new Date()
              };

              setMessages([...updatedMessages, errorMessage]);
            }
          }
        }
      }

      // If we reach here without getting a 'done' signal, mark as complete
      setMessages(prev => prev.map(msg => 
        msg.id === aiMessageId 
          ? { ...msg, isStreaming: false }
          : msg
      ));

    } catch (error) {
      console.error('Error sending message:', error);
      
      // Check if it was aborted
      if (error.name === 'AbortError') {
        // Update the message to show it was stopped
        setMessages(prev => prev.map(msg => 
          msg.id === aiMessageId 
            ? { ...msg, text: accumulatedText || 'Response stopped by user.', isStreaming: false }
            : msg
        ));
      } else {
        console.error('Error uploading file:', error);
      
        const errorMessage = {
          id: Date.now(),
          text: `❌ Sorry, I encountered an error while sending response to server. Please try again.`,
          sender: 'varuna',
          timestamp: new Date()
        };

        setMessages([...messages, errorMessage]);
      }
    } finally {
      setIsTyping(false);
      setIsThinking(false);
      setAbortController(null);
    }
  };

  const handleStopGeneration = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsTyping(false);
    }
  };

  const handleClearChat = async () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    if (isThinking || isTyping){
        handleStopGeneration();
    }
    setMessages([getInitialMessage()]);
    setInputText('');
    setIsTyping(false);
    setIsThinking(false)
    // Clear RAG documents if session exists
    if (ragSessionId && hasUploadedDocuments) {
      await deleteRagSession();
    }
  };

  const handleFileAttach = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    console.log('File selected:', file.name);

    // Validate file type
    const allowedTypes = ['.pdf', '.txt', '.docx', '.doc','.png','.jgp','jpeg'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExtension)) {
      alert(`Unsupported file type. Please upload: ${allowedTypes.join(', ')}`);
      return;
    }

    // Ensure we have a RAG session
    let sessionId = ragSessionId;
    if (!sessionId) {
      sessionId = await createRagSession();
      if (!sessionId) {
        alert('Failed to create RAG session. Please try again.');
        return;
      }
    }

    // Set typing state to show loading
    setIsTyping(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${BACKEND_PATH}/rag/upload/${sessionId}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setHasUploadedDocuments(true);
        setUploadedFileName(file.name);
      } else {
        throw new Error(data.message || 'Upload failed');
      }
      
    } catch (error) {
      console.error('Error uploading file:', error);
      
      const errorMessage = {
        id: Date.now(),
        text: `❌ Sorry, I encountered an error while uploading the file: ${error.message}. Please try again.`,
        sender: 'varuna',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Handle logout with backend call
  const handleLogout = async () => {
    try {
      setIsLoggingOut(true);
      
      // Call the backend logout endpoint
      const response = await fetch(`${BACKEND_PATH}/logout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: user.name
        })
      });

      if (response.ok) {
        console.log('Successfully logged out from backend');
        onLogout();
      } else {
        const errorData = await response.json();
        console.error('Logout failed:', errorData);
        
        onLogout();
      }
    } catch (error) {
      console.error('Error during logout:', error);
      onLogout();
    } finally {
      setIsLoggingOut(false);
    }
  };


  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-content">
          <div className="header-left">
            <Anchor size={20} style={{ marginRight: '8px', verticalAlign: 'middle' , color: 'white'}} />
            <h1>Varuna AI {hasUploadedDocuments && <span className="rag-indicator"> RAG</span>}</h1>
          </div>
          <div className="user-info">
            <button 
              onClick={handleClearChat}
              className="clear-chat-button header-toggle"
              title="Clear chat history"
            >
              <Trash2 size={16} style={{color: 'white'}} />
            </button>
            <button 
              onClick={toggleDarkMode}
              className="dark-mode-toggle header-toggle"
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <span className="user-name">
              <User size={16} />
              {user?.name}
            </span>
            <button 
              onClick={handleLogout} 
              className="logout-button"
              disabled={isLoggingOut}
            >
            <LogOut size={16} />
            {isLoggingOut ? 'Logging out...' : 'Logout'}
            </button>
          </div>
        </div>
      </header>

      <div className="chat-messages">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {isThinking && (
          <div className="typing-indicator">
            <div className="typing-message">
              <div className="avatar varuna-avatar">V</div>
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Thinking indicator */}
      {isTyping && (
        <div className="thinking-indicator">
          <div className="thinking-content">
            <Bot size={16} className="thinking-icon" />
            <span className="thinking-text">Varuna is thinking</span>
            <div className="thinking-dots">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        </div>
      )}


      <form onSubmit={handleSendMessage} className="chat-input-form">
        <div className="input-container">
          {/* File indicator above chat input */}
          {hasUploadedDocuments && uploadedFileName && (
            <div className="file-indicator">
              <div className={`file-item ${uploadedFileName.split('.').pop().toLowerCase()}`}>
                <FileText size={16} className="file-icon" />
                <span className="file-name">{uploadedFileName}</span>
                <button
                  onClick={deleteRagSession}
                  className="file-delete-button"
                  title="Remove uploaded file"
                  disabled={isThinking}
                >
                  <X size={14} />
                </button>
              </div>
            </div>
          )}
          
          <div className="input-bar">
            <button
              type="button"
              onClick={handleFileAttach}
              className="attach-button"
              title="Attach file for RAG processing (PDF, TXT, DOCX)"
              disabled={isTyping || isThinking}
            >
              <Paperclip size={20} />
            </button>
            <textarea
              value={inputText}
              onChange={(e) => {
                setInputText(e.target.value);
                adjustTextareaHeight(e.target);
              }}
              placeholder={`Message Varuna (${models[currentModel].name})${hasUploadedDocuments ? ' - RAG enabled' : ''}`}
              className="chat-input"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!isTyping && inputText.trim()) {
                    handleSendMessage(e);
                  }
                }
              }}
              rows={1}
              style={{ overflow: 'hidden', maxHeight: '160px' }} // Approx. 6 rows
              ref={textareaRef}
            />

            <button
              type="button"
              className={`send-button ${isThinking ? 'stop-button' : ''}`}
              disabled={!isTyping && !isThinking}
              title={isThinking ? "Stop generation" : "Send message"}
              onClick={isThinking ? handleStopGeneration : handleSendMessage}
            >
              {isThinking ? <Square size={20} /> : <Send size={20} />}
            </button>
          </div>
        </div>
        
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          accept=".pdf,.txt,.docx,.doc,.png,.jpeg,.jpg"
        />
      </form>
    </div>
  );
};

export default ChatInterface;