import { useState } from 'react';
import './App.css';
import Chatbot from './components/chatbot/Chatbot';

export default function App() {
  return (
    <> 
      <h1 id="welcome_text"> How can I help you today? </h1>

      <Chatbot />
    </>
  )
}
