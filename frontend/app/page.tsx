'use client'
import { useState, useRef, useEffect } from 'react';
import { Send, Brain, AlertCircle, Sparkles, MessageCircle, Trash2, Moon, Sun, BarChart3, History, TrendingUp, Calendar, Clock } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar } from 'recharts';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  sentiment?: string;
  isCrisis?: boolean;
  timestamp: Date;
}

interface Session {
  id: string;
  startTime: Date;
  endTime: Date;
  messages: Message[];
  dominantSentiment: string;
  messageCount: number;
}

interface MoodEntry {
  date: string;
  sentiment: string;
  count: number;
  timestamp: number;
}

const SENTIMENT_CONFIGS: Record<string, { color: string; emoji: string; gradient: string; chartColor: string }> = {
  'Anxiety': { 
    color: 'bg-amber-100 text-amber-800 border-amber-300',
    emoji: '😰',
    gradient: 'from-amber-400 to-orange-500',
    chartColor: '#f59e0b'
  },
  'Depression': { 
    color: 'bg-blue-100 text-blue-800 border-blue-300',
    emoji: '😔',
    gradient: 'from-blue-400 to-indigo-500',
    chartColor: '#3b82f6'
  },
  'Suicidal': { 
    color: 'bg-red-100 text-red-800 border-red-300',
    emoji: '🆘',
    gradient: 'from-red-500 to-rose-600',
    chartColor: '#ef4444'
  },
  'Stress': { 
    color: 'bg-orange-100 text-orange-800 border-orange-300',
    emoji: '😫',
    gradient: 'from-orange-400 to-amber-500',
    chartColor: '#f97316'
  },
  'Bipolar': { 
    color: 'bg-purple-100 text-purple-800 border-purple-300',
    emoji: '🎭',
    gradient: 'from-purple-400 to-pink-500',
    chartColor: '#a855f7'
  },
  'Emotional Intensity': { 
    color: 'bg-pink-100 text-pink-800 border-pink-300',
    emoji: '💗',
    gradient: 'from-pink-400 to-rose-500',
    chartColor: '#ec4899'
  },
  'Normal': { 
    color: 'bg-green-100 text-green-800 border-green-300',
    emoji: '😊',
    gradient: 'from-green-400 to-emerald-500',
    chartColor: '#10b981'
  },
};

export default function ChatPage() {
  const [currentView, setCurrentView] = useState<'chat' | 'dashboard' | 'history'>('chat');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      content: "Hello, I'm here to listen and support you. This is a safe space where you can share what's on your mind. How are you feeling today?",
      timestamp: new Date(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [moodData, setMoodData] = useState<MoodEntry[]>([]);
  const [currentSessionStart, setCurrentSessionStart] = useState<Date>(new Date());
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [confirmModalData, setConfirmModalData] = useState<{
    message: string;
    onConfirm: () => void;
  } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  // Load data on mount
  useEffect(() => {
    loadSessionHistory();
    loadMoodData();
  }, []);

  // Track if we've already saved mood data for a message
  const savedMoodIds = useRef<Set<string>>(new Set());

  const loadSessionHistory = () => {
    try {
      const stored = localStorage.getItem('ai-therapist-sessions');
      if (stored) {
        const loadedSessions = JSON.parse(stored).map((s: any) => ({
          ...s,
          startTime: new Date(s.startTime),
          endTime: new Date(s.endTime),
          messages: s.messages.map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          }))
        }));
        setSessions(loadedSessions.sort((a: Session, b: Session) => b.startTime.getTime() - a.startTime.getTime()));
      }
    } catch (error) {
      console.error('Failed to load session history:', error);
    }
  };

  const loadMoodData = () => {
    try {
      const stored = localStorage.getItem('ai-therapist-mood-data');
      if (stored) {
        setMoodData(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load mood data:', error);
    }
  };

  const saveMoodData = (sentimentMessages: Message[]) => {
    // Only save NEW mood entries (not already saved)
    const newEntries: MoodEntry[] = sentimentMessages
      .filter(m => !savedMoodIds.current.has(m.id))
      .map(m => {
        savedMoodIds.current.add(m.id); // Mark as saved
        return {
          date: m.timestamp.toLocaleDateString(),
          sentiment: m.sentiment!,
          count: 1,
          timestamp: m.timestamp.getTime()
        };
      });

    if (newEntries.length === 0) return;

    try {
      const stored = localStorage.getItem('ai-therapist-mood-data');
      const existing = stored ? JSON.parse(stored) : [];
      const allEntries = [...existing, ...newEntries];
      localStorage.setItem('ai-therapist-mood-data', JSON.stringify(allEntries));
      setMoodData(allEntries);
    } catch (error) {
      console.error('Failed to save mood data:', error);
    }
  };

  const saveSession = (messagesToSave?: Message[]) => {
    const messagesToUse = messagesToSave || messages;
    if (messagesToUse.length <= 1) return;

    const sentiments = messagesToUse.filter(m => m.sentiment).map(m => m.sentiment!);
    const sentimentCounts = sentiments.reduce((acc, s) => {
      acc[s] = (acc[s] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    const dominantSentiment = Object.entries(sentimentCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Normal';

    const session: Session = {
      id: Date.now().toString(),
      startTime: currentSessionStart,
      endTime: new Date(),
      messages: messagesToUse,
      dominantSentiment,
      messageCount: messagesToUse.length
    };

    try {
      const stored = localStorage.getItem('ai-therapist-sessions');
      const existing = stored ? JSON.parse(stored) : [];
      const updated = [session, ...existing];
      localStorage.setItem('ai-therapist-sessions', JSON.stringify(updated));
      loadSessionHistory();
    } catch (error) {
      console.error('Failed to save session:', error);
    }
  };

  const clearChat = () => {
    // Capture current messages to ensure we save them before clearing
    const currentMessages = messages;
    setConfirmModalData({
      message: 'Are you sure you want to clear the conversation? This will save the current session.',
      onConfirm: () => {
        saveSession(currentMessages);
        setMessages([{
          id: Date.now().toString(),
          type: 'bot',
          content: "Hello again. I'm here whenever you need to talk. What's on your mind?",
          timestamp: new Date(),
        }]);
        setCurrentSessionStart(new Date());
        setShowConfirmModal(false);
        setConfirmModalData(null);
      }
    });
    setShowConfirmModal(true);
  };

  const loadSession = (session: Session) => {
    setMessages(session.messages);
    setCurrentView('chat');
  };

  const deleteSession = (sessionId: string) => {
    setConfirmModalData({
      message: 'Are you sure you want to delete this session?',
      onConfirm: () => {
        try {
          const stored = localStorage.getItem('ai-therapist-sessions');
          if (stored) {
            const sessions = JSON.parse(stored);
            const updated = sessions.filter((s: Session) => s.id !== sessionId);
            localStorage.setItem('ai-therapist-sessions', JSON.stringify(updated));
            loadSessionHistory();
          }
        } catch (error) {
          console.error('Failed to delete session:', error);
        }
        setShowConfirmModal(false);
        setConfirmModalData(null);
      }
    });
    setShowConfirmModal(true);
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setIsTyping(true);
    setError('');

    // Build conversation history properly - exclude the initial greeting
    const history = messages
      .slice(1) // Skip the first bot greeting message
      .map(msg => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.content
      }));

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: input,
          conversation_history: history 
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      setTimeout(() => {
        setIsTyping(false);
        const botMessage: Message = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.response,
          sentiment: data.sentiment, 
          isCrisis: data.is_crisis ,
          timestamp: new Date(),
        };
        setMessages(prev => {
          const updated = [...prev, botMessage];
          // Save mood data for this new message only
          if (botMessage.sentiment) {
            saveMoodData([botMessage]);
          }
          return updated;
        });
      }, 800);

    } catch (err) {
      setIsTyping(false);
      setError('Unable to connect. Please check if the backend is running.');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const getMoodTrendData = () => {
    const last7Days = Array.from({length: 7}, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (6 - i));
      return date.toLocaleDateString();
    });

    return last7Days.map(date => {
      const dayMoods = moodData.filter(m => m.date === date);
      const sentimentCounts = dayMoods.reduce((acc, m) => {
        acc[m.sentiment] = (acc[m.sentiment] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);

      const dominant = Object.entries(sentimentCounts).sort((a, b) => b[1] - a[1])[0];
      return {
        date: date.split('/').slice(0, 2).join('/'),
        sentiment: dominant?.[0] || 'Normal',
        value: dominant?.[1] || 0
      };
    });
  };

  const getSentimentDistribution = () => {
    const counts = moodData.reduce((acc, m) => {
      acc[m.sentiment] = (acc[m.sentiment] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(counts).map(([name, value]) => ({
      name,
      value,
      color: SENTIMENT_CONFIGS[name]?.chartColor || '#gray'
    }));
  };

  const bgClass = darkMode 
    ? 'bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900' 
    : 'bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50';

  const cardBg = darkMode ? 'bg-slate-800/90' : 'bg-white/90';
  const textColor = darkMode ? 'text-white' : 'text-gray-800';

  return (
    <div className={`min-h-screen ${bgClass} transition-colors duration-300`}>
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className={`absolute top-20 left-10 w-72 h-72 ${darkMode ? 'bg-purple-500' : 'bg-purple-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob`}></div>
        <div className={`absolute top-40 right-10 w-72 h-72 ${darkMode ? 'bg-pink-500' : 'bg-pink-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000`}></div>
        <div className={`absolute -bottom-8 left-1/2 w-72 h-72 ${darkMode ? 'bg-blue-500' : 'bg-blue-300'} rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000`}></div>
      </div>

      {/* Header */}
      <header className={`${cardBg} backdrop-blur-xl shadow-lg border-b ${darkMode ? 'border-purple-700' : 'border-purple-100'} relative z-10`}>
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl blur opacity-75 animate-pulse"></div>
                <div className="relative bg-gradient-to-br from-purple-500 to-pink-500 p-2.5 rounded-xl">
                  <Brain className="w-7 h-7 text-white" />
                </div>
              </div>
              <div>
                <h1 className={`text-xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent ${darkMode && 'from-purple-400 to-pink-400'}`}>
                  MindfulAI Therapist
                </h1>
                <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Your compassionate companion</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className={`p-2 rounded-lg ${darkMode ? 'bg-slate-700 text-yellow-400' : 'bg-purple-100 text-purple-600'} hover:scale-110 transition-transform`}
              >
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
              {currentView === 'chat' && (
                <button
                  onClick={clearChat}
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg ${darkMode ? 'bg-slate-700 text-red-400 hover:bg-slate-600' : 'bg-red-50 text-red-600 hover:bg-red-100'} transition-colors text-sm`}
                >
                  <Trash2 className="w-4 h-4" />
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex gap-2">
            <button
              onClick={() => setCurrentView('chat')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                currentView === 'chat'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                  : darkMode ? 'bg-slate-700/50 text-gray-300 hover:bg-slate-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <MessageCircle className="w-4 h-4" />
              <span className="text-sm font-medium">Chat</span>
            </button>
            <button
              onClick={() => setCurrentView('dashboard')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                currentView === 'dashboard'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                  : darkMode ? 'bg-slate-700/50 text-gray-300 hover:bg-slate-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <BarChart3 className="w-4 h-4" />
              <span className="text-sm font-medium">Dashboard</span>
            </button>
            <button
              onClick={() => setCurrentView('history')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                currentView === 'history'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                  : darkMode ? 'bg-slate-700/50 text-gray-300 hover:bg-slate-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <History className="w-4 h-4" />
              <span className="text-sm font-medium">History</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Crisis Banner */}
      {currentView === 'chat' && (
        <div className="max-w-6xl mx-auto px-4 py-3 relative z-10">
          <div className={`${darkMode ? 'bg-red-900/50 border-red-700' : 'bg-red-50 border-red-200'} border rounded-xl p-3 flex items-start gap-2 backdrop-blur-sm`}>
            <AlertCircle className={`w-5 h-5 ${darkMode ? 'text-red-400' : 'text-red-600'} shrink-0 mt-0.5`} />
            <div className={`text-sm ${darkMode ? 'text-red-300' : 'text-red-800'}`}>
              <strong>Crisis Support:</strong> If you're in immediate danger, please call emergency services or contact a crisis helpline immediately.
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 pb-4 relative z-10">
        {currentView === 'chat' && (
          <div className={`${cardBg} backdrop-blur-xl rounded-3xl shadow-2xl border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} flex flex-col h-[calc(100vh-280px)]`}>
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
                >
                  <div className={`max-w-[85%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                    {message.sentiment && (
                      <div className="mb-2 flex items-center gap-2">
                        <span className={`text-xs px-3 py-1.5 rounded-full border backdrop-blur-sm ${SENTIMENT_CONFIGS[message.sentiment]?.color || 'bg-gray-100 text-gray-800'}`}>
                          <span className="mr-1">{SENTIMENT_CONFIGS[message.sentiment]?.emoji}</span>
                          {message.sentiment}
                        </span>
                      </div>
                    )}
                    <div
                      className={`rounded-2xl px-5 py-3.5 shadow-lg ${
                        message.type === 'user'
                          ? `bg-gradient-to-br ${SENTIMENT_CONFIGS[message.sentiment || 'Normal']?.gradient || 'from-purple-500 to-pink-500'} text-white`
                          : message.isCrisis
                          ? `${darkMode ? 'bg-red-900/50 border-2 border-red-500' : 'bg-red-50 border-2 border-red-300'} ${darkMode ? 'text-red-200' : 'text-gray-800'}`
                          : `${darkMode ? 'bg-slate-700/80' : 'bg-gray-50'} ${textColor}`
                      }`}
                    >
                      {message.type === 'bot' && !message.isCrisis && (
                        <div className="flex items-center gap-2 mb-2 opacity-70">
                          <Sparkles className="w-3.5 h-3.5" />
                          <span className="text-xs font-medium">AI Therapist</span>
                        </div>
                      )}
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                    </div>
                    <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'} mt-1.5 px-2`}>
                      {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex justify-start">
                  <div className={`${darkMode ? 'bg-slate-700/80' : 'bg-gray-50'} rounded-2xl px-5 py-3.5 flex items-center gap-2 shadow-lg`}>
                    <div className="flex gap-1">
                      <div className={`w-2 h-2 ${darkMode ? 'bg-purple-400' : 'bg-purple-500'} rounded-full animate-bounce`}></div>
                      <div className={`w-2 h-2 ${darkMode ? 'bg-purple-400' : 'bg-purple-500'} rounded-full animate-bounce`} style={{ animationDelay: '0.2s' }}></div>
                      <div className={`w-2 h-2 ${darkMode ? 'bg-purple-400' : 'bg-purple-500'} rounded-full animate-bounce`} style={{ animationDelay: '0.4s' }}></div>
                    </div>
                    <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Listening...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {error && (
              <div className="px-6 pb-2">
                <div className={`${darkMode ? 'bg-red-900/50 border-red-700' : 'bg-red-50 border-red-200'} border rounded-xl p-3 text-sm ${darkMode ? 'text-red-300' : 'text-red-700'}`}>
                  {error}
                </div>
              </div>
            )}

            <div className={`border-t ${darkMode ? 'border-purple-700/50' : 'border-gray-200'} p-4`}>
              <div className="flex gap-3 items-end">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Share what's on your mind..."
                  className={`flex-1 resize-none rounded-2xl border ${darkMode ? 'border-purple-700/50 bg-slate-700/50 text-white placeholder-gray-500' : 'border-gray-300 bg-white text-gray-800 placeholder-gray-500'} px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 text-sm max-h-32`}
                  rows={1}
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!input.trim() || isLoading}
                  className="bg-gradient-to-br from-purple-500 to-pink-500 text-white p-4 rounded-2xl hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:scale-105 transition-all"
                >
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {currentView === 'dashboard' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`p-3 rounded-xl ${darkMode ? 'bg-purple-500/20' : 'bg-purple-100'}`}>
                    <MessageCircle className={`w-6 h-6 ${darkMode ? 'text-purple-400' : 'text-purple-600'}`} />
                  </div>
                  <div>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Total Sessions</p>
                    <p className={`text-2xl font-bold ${textColor}`}>{sessions.length}</p>
                  </div>
                </div>
              </div>

              <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`p-3 rounded-xl ${darkMode ? 'bg-pink-500/20' : 'bg-pink-100'}`}>
                    <TrendingUp className={`w-6 h-6 ${darkMode ? 'text-pink-400' : 'text-pink-600'}`} />
                  </div>
                  <div>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Mood Entries</p>
                    <p className={`text-2xl font-bold ${textColor}`}>{moodData.length}</p>
                  </div>
                </div>
              </div>

              <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`p-3 rounded-xl ${darkMode ? 'bg-blue-500/20' : 'bg-blue-100'}`}>
                    <Calendar className={`w-6 h-6 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                  </div>
                  <div>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Days Tracked</p>
                    <p className={`text-2xl font-bold ${textColor}`}>{new Set(moodData.map(m => m.date)).size}</p>
                  </div>
                </div>
              </div>
            </div>

            <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
              <h3 className={`text-lg font-semibold mb-4 ${textColor}`}>7-Day Mood Trend</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={getMoodTrendData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                  <XAxis dataKey="date" stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                  <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: darkMode ? '#1e293b' : '#fff',
                      border: `1px solid ${darkMode ? '#4c1d95' : '#e5e7eb'}`,
                      borderRadius: '8px'
                    }}
                    formatter={(value: any, name: string) => [value, 'Occurrences']}
                  />
                  <Line type="monotone" dataKey="value" stroke="#a855f7" strokeWidth={3} dot={{ fill: '#a855f7', r: 6 }} name="Mood Count" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-4 ${textColor}`}>Sentiment Distribution</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={getSentimentDistribution()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={(entry: any) => `${entry.name} ${(entry.percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {getSentimentDistribution().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: darkMode ? '#1e293b' : '#fff',
                        border: `1px solid ${darkMode ? '#4c1d95' : '#e5e7eb'}`,
                        borderRadius: '8px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
                <h3 className={`text-lg font-semibold mb-4 ${textColor}`}>Sentiment Frequency</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getSentimentDistribution()}>
                    <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#374151' : '#e5e7eb'} />
                    <XAxis dataKey="name" stroke={darkMode ? '#9ca3af' : '#6b7280'} angle={-45} textAnchor="end" height={100} />
                    <YAxis stroke={darkMode ? '#9ca3af' : '#6b7280'} />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: darkMode ? '#1e293b' : '#fff',
                        border: `1px solid ${darkMode ? '#4c1d95' : '#e5e7eb'}`,
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="value" fill="#a855f7" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
              <h3 className={`text-lg font-semibold mb-4 ${textColor}`}>Recent Mood Insights</h3>
              <div className="space-y-3">
                {moodData.slice(-5).reverse().map((entry, idx) => (
                  <div key={idx} className={`flex items-center justify-between p-3 rounded-lg ${darkMode ? 'bg-slate-700/50' : 'bg-gray-50'}`}>
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">{SENTIMENT_CONFIGS[entry.sentiment]?.emoji}</span>
                      <div>
                        <p className={`font-medium ${textColor}`}>{entry.sentiment}</p>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{entry.date}</p>
                      </div>
                    </div>
                    <span className={`text-xs px-3 py-1 rounded-full ${SENTIMENT_CONFIGS[entry.sentiment]?.color}`}>
                      Tracked
                    </span>
                  </div>
                ))}
                {moodData.length === 0 && (
                  <p className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    No mood data yet. Start chatting to track your emotional journey!
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {currentView === 'history' && (
          <div className="space-y-4">
            <div className={`${cardBg} backdrop-blur-xl rounded-2xl p-6 border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} shadow-lg`}>
              <h2 className={`text-2xl font-bold mb-6 ${textColor}`}>Session History</h2>
              
              {sessions.length === 0 ? (
                <div className="text-center py-12">
                  <History className={`w-16 h-16 mx-auto mb-4 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
                  <p className={`text-lg ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>No sessions yet</p>
                  <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'} mt-2`}>Your conversation history will appear here</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {sessions.map((session) => (
                    <div
                      key={session.id}
                      className={`border ${darkMode ? 'border-purple-700/50 bg-slate-700/50' : 'border-purple-100 bg-gray-50'} rounded-xl p-5 hover:shadow-lg transition-all cursor-pointer`}
                    >
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${darkMode ? 'bg-purple-500/20' : 'bg-purple-100'}`}>
                            <MessageCircle className={`w-5 h-5 ${darkMode ? 'text-purple-400' : 'text-purple-600'}`} />
                          </div>
                          <div>
                            <h3 className={`font-semibold ${textColor}`}>
                              Session on {session.startTime.toLocaleDateString()}
                            </h3>
                            <div className="flex items-center gap-3 mt-1">
                              <span className={`text-xs flex items-center gap-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                <Clock className="w-3 h-3" />
                                {session.startTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - 
                                {session.endTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                              </span>
                              <span className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                {session.messageCount} messages
                              </span>
                            </div>
                          </div>
                        </div>
                        <span className={`text-xs px-3 py-1.5 rounded-full ${SENTIMENT_CONFIGS[session.dominantSentiment]?.color}`}>
                          <span className="mr-1">{SENTIMENT_CONFIGS[session.dominantSentiment]?.emoji}</span>
                          {session.dominantSentiment}
                        </span>
                      </div>

                      <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'} mb-3 line-clamp-2`}>
                        {session.messages.filter(m => m.type === 'user')[0]?.content || 'No messages'}
                      </div>

                      <div className="flex gap-2">
                        <button
                          onClick={() => loadSession(session)}
                          className={`flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            darkMode 
                              ? 'bg-purple-500/20 text-purple-400 hover:bg-purple-500/30' 
                              : 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                          }`}
                        >
                          View Session
                        </button>
                        <button
                          onClick={() => deleteSession(session.id)}
                          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                            darkMode 
                              ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' 
                              : 'bg-red-100 text-red-700 hover:bg-red-200'
                          }`}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Confirmation Modal */}
      {showConfirmModal && confirmModalData && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <div 
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => {
              setShowConfirmModal(false);
              setConfirmModalData(null);
            }}
          />
          
          {/* Modal */}
          <div className={`relative ${cardBg} backdrop-blur-xl rounded-3xl shadow-2xl border ${darkMode ? 'border-purple-700/50' : 'border-purple-100'} p-6 max-w-md w-full animate-fadeIn`}>
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-2 rounded-xl ${darkMode ? 'bg-red-500/20' : 'bg-red-100'}`}>
                <AlertCircle className={`w-6 h-6 ${darkMode ? 'text-red-400' : 'text-red-600'}`} />
              </div>
              <h3 className={`text-xl font-semibold ${textColor}`}>Confirm Action</h3>
            </div>
            
            <p className={`text-sm mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              {confirmModalData.message}
            </p>
            
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => {
                  setShowConfirmModal(false);
                  setConfirmModalData(null);
                }}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  darkMode 
                    ? 'bg-slate-700 text-gray-300 hover:bg-slate-600' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                Cancel
              </button>
              <button
                onClick={confirmModalData.onConfirm}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all bg-gradient-to-r from-red-500 to-red-600 text-white hover:from-red-600 hover:to-red-700 shadow-lg`}
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes blob {
          0%, 100% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
        }
        .animate-fadeIn { animation: fadeIn 0.5s ease-out; }
        .animate-blob { animation: blob 7s infinite; }
        .animation-delay-2000 { animation-delay: 2s; }
        .animation-delay-4000 { animation-delay: 4s; }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
}