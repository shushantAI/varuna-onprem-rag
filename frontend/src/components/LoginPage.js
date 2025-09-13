import React, { useState } from 'react';
import { User, Lock, Moon, Sun, UserPlus } from 'lucide-react';

const BACKEND_PATH='http://localhost:8000'

const LoginPage = ({ onLogin, darkMode, toggleDarkMode }) => {
  const [isRegisterMode, setIsRegisterMode] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.password.trim()) {
      newErrors.password = 'Password is required';
    }
    
    return newErrors;
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    const newErrors = validateForm();
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch(BACKEND_PATH+'/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: formData.name,
          password: formData.password
        }),
      });
      const data = await response.json();
      if (response.ok) {
        console.log('Login successful:', data);
        onLogin({
          name: data.username,
          message: data.message
        });
      } else {
        // Check if status code is 401 and show popup
        if (response.status === 401) {
          alert('Login failed');
        }
        
        // Handle login errors
        setErrors({
          general: data.detail || 'Login failed. Please check your credentials.'
        });
      }
    } catch (error) {
      console.error('Login error:', error);
      setErrors({
        general: 'Network error. Please try again later.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    const newErrors = validateForm();
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    setIsLoading(true);
    try {
      const response = await fetch(BACKEND_PATH+'/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: formData.name,
          password: formData.password
        }),
      });
      const data = await response.json();
      if (response.ok) {
        console.log('Registration successful:', data);
        // Switch back to login mode and show success message
        setIsRegisterMode(false);
        setSuccessMessage('Registered successfully! Please log in.');
        setFormData({ name: '', password: '' });
        setErrors({});
      } else {
        // Handle registration errors
        if (response.status === 409 || data.detail?.includes('already')) {
          setErrors({
            general: 'Username already in use'
          });
        } else {
          setErrors({
            general: data.detail || 'Registration failed. Please try again.'
          });
        }
      }
    } catch (error) {
      console.error('Registration error:', error);
      setErrors({
        general: 'Network error. Please try again later.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const switchMode = () => {
    setIsRegisterMode(!isRegisterMode);
    setFormData({ name: '', password: '' });
    setErrors({});
    setSuccessMessage('');
  };

  return (
    <div className="login-container">
      <button 
        onClick={toggleDarkMode}
        className="dark-mode-toggle"
        title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {darkMode ? <Sun size={20} /> : <Moon size={20} />}
      </button>
      
      <div className="login-card">
        <div className="login-header">
          <h1>{isRegisterMode ? 'Create Account' : 'Welcome to Varuna AI'}</h1>
          <p>{isRegisterMode ? 'Please enter your details to create an account' : 'Please enter your details to continue'}</p>
        </div>
        
        {successMessage && (
          <div className="success-message">
            {successMessage}
          </div>
        )}
        
        <form onSubmit={isRegisterMode ? handleRegister : handleLogin} className="login-form">
          <div className="form-group">
            <label htmlFor="name">
              <User size={18} />
              User Name
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              className={errors.name ? 'error' : ''}
              placeholder="Enter your full name"
            />
            {errors.name && <span className="error-message">{errors.name}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="password">
              <Lock size={18} />
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              className={errors.password ? 'error' : ''}
              placeholder="Enter your password"
            />
            {errors.password && <span className="error-message">{errors.password}</span>}
          </div>

          {errors.general && (
            <div className="error-message general-error">
              {errors.general}
            </div>
          )}

          <button 
            type="submit" 
            className="login-button"
            disabled={isLoading}
          >
            {isLoading ? 
              (isRegisterMode ? 'Creating Account...' : 'Logging in...') : 
              (isRegisterMode ? 'Create Account' : 'Login')
            }
          </button>
        </form>

        <div className="auth-switch">
          <p>
            {isRegisterMode ? 'Already have an account?' : "Don't have an account?"}
          </p>
          <button 
            type="button" 
            className="login-button"
            onClick={switchMode}
            disabled={isLoading}
          >
            <UserPlus size={16} />
            {isRegisterMode ? 'Login' : 'Register'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;