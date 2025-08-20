import { useState } from 'react';
import './App.css';
import Chatbot from './components/chatbot/Chatbot';

export default function App() {
  return (
    <> 
      <h1 id="welcome_text"> What do you want to do today? </h1>

      <Chatbot />
    </>
  )
}
