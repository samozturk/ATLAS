'use client';

import {
  type KeyboardEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from 'react';
import {
  Activity,
  ArrowUp,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleDot,
  CloudSun,
  Command,
  Clock3,
  Cpu,
  Gauge,
  History,
  Home,
  MessageSquare,
  MoreHorizontal,
  PanelLeft,
  Plus,
  Radio,
  Settings2,
  Sparkles,
  Waves,
  Wrench,
  Zap,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

type Brain = 'fast' | 'deep';

type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  brain?: Brain;
  model?: string;
  isStreaming?: boolean;
  toolActivity?: ToolActivity[];
};

type ToolActivity = {
  id: string;
  name: string;
  location?: string;
  ok?: boolean;
  content?: string;
};

type ConversationSummary = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

type StoredMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  brain?: Brain;
  model?: string;
  tool_activity?: Array<{
    tool_call_id?: string;
    tool_name: string;
    arguments?: Record<string, unknown>;
    ok?: boolean;
    content?: string;
  }>;
};

type StoredConversation = ConversationSummary & { messages: StoredMessage[] };

type ToolCallPayload = {
  id?: string;
  function?: {
    name?: string;
    arguments?: Record<string, unknown>;
  };
};

type ToolResultPayload = {
  tool_call_id?: string;
  tool_name?: string;
  ok?: boolean;
  content?: string;
};

type StreamPayload = {
  type: 'thinking' | 'token' | 'tool_call' | 'done';
  content?: string;
  brain?: Brain;
  model?: string;
  detail?: string;
  tool_calls?: ToolCallPayload[];
  tool_results?: ToolResultPayload[];
};

const brains: Record<Brain, { label: string; model: string; description: string; accent: string }> = {
  fast: {
    label: 'Fast brain',
    model: 'Qwen 3.6 · 35B',
    description: 'Everyday conversation, quick decisions, and responsive control.',
    accent: 'cyan',
  },
  deep: {
    label: 'Deep brain',
    model: 'Qwen 3.5 · 122B-A10B',
    description: 'Deliberate reasoning for complex, multi-step questions.',
    accent: 'violet',
  },
};

const initialMessages: Message[] = [
  {
    id: 'atlas-intro',
    role: 'assistant',
    brain: 'fast',
    model: 'ATLAS local runtime',
    content:
      'I’m ready. Ask me something, choose a brain, or let’s work through a problem together.',
  },
];

function streamFrames(chunk: string): Array<{ event: string; data: StreamPayload }> {
  return chunk
    .split('\n\n')
    .filter(Boolean)
    .flatMap((frame) => {
      const event = frame
        .split('\n')
        .find((line) => line.startsWith('event:'))
        ?.slice('event:'.length)
        .trim();
      const rawData = frame
        .split('\n')
        .find((line) => line.startsWith('data:'))
        ?.slice('data:'.length)
        .trim();
      if (!event || !rawData) return [];
      try {
        return [{ event, data: JSON.parse(rawData) as StreamPayload }];
      } catch {
        return [];
      }
    });
}

function markAssistantUnavailable(message: Message, assistantId: string, detail: string): Message {
  if (message.id !== assistantId) return message;
  return {
    ...message,
    content: `I couldn’t reach the local runtime. ${detail}`,
    isStreaming: false,
  };
}

function markConversationUnavailable(assistantId: string, detail: string) {
  return (current: Message[]) =>
    current.map((message) => markAssistantUnavailable(message, assistantId, detail));
}

function createMessageId(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') {
    return globalThis.crypto.randomUUID();
  }

  return `message-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function toolActivityFromCalls(toolCalls: ToolCallPayload[]): ToolActivity[] {
  return toolCalls.flatMap((toolCall, index) => {
    const name = toolCall.function?.name;
    if (!name) return [];
    const location = toolCall.function?.arguments?.location;
    return [{
      id: toolCall.id ?? `${name}-${index}`,
      name,
      location: typeof location === 'string' ? location : undefined,
    }];
  });
}

function toolActivityFromStored(activity: StoredMessage['tool_activity']): ToolActivity[] {
  return (activity ?? []).map((tool, index) => ({
    id: tool.tool_call_id ?? `${tool.tool_name}-${index}`,
    name: tool.tool_name,
    location: typeof tool.arguments?.location === 'string' ? tool.arguments.location : undefined,
    ok: tool.ok,
    content: tool.content,
  }));
}

function applyToolResults(activity: ToolActivity[], results: ToolResultPayload[]): ToolActivity[] {
  return activity.map((tool) => {
    const result = results.find((candidate) =>
      candidate.tool_call_id ? candidate.tool_call_id === tool.id : candidate.tool_name === tool.name && tool.ok === undefined,
    );
    return result ? { ...tool, ok: result.ok, content: result.content } : tool;
  });
}

function messageFromStored(message: StoredMessage): Message {
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    brain: message.brain,
    model: message.model,
    toolActivity: toolActivityFromStored(message.tool_activity),
  };
}

export default function HomePage() {
  const [brain, setBrain] = useState<Brain>('fast');
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [draft, setDraft] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [connection, setConnection] = useState<'ready' | 'working' | 'offline'>('ready');
  const transcriptEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, isStreaming]);

  const loadConversation = async (id: string) => {
    const response = await fetch(`/conversations/${id}`);
    if (!response.ok) throw new Error('ATLAS could not load this conversation.');
    const conversation = await response.json() as StoredConversation;
    setConversationId(conversation.id);
    setMessages(conversation.messages.length ? conversation.messages.map(messageFromStored) : initialMessages);
  };

  const refreshConversations = async () => {
    const response = await fetch('/conversations');
    if (!response.ok) throw new Error('ATLAS could not load conversation history.');
    const data = await response.json() as ConversationSummary[];
    setConversations(data);
    return data;
  };

  const createConversation = async () => {
    const response = await fetch('/conversations', { method: 'POST' });
    if (!response.ok) throw new Error('ATLAS could not start a new conversation.');
    const conversation = await response.json() as ConversationSummary;
    setConversationId(conversation.id);
    setMessages(initialMessages);
    setConversations((current) => [conversation, ...current]);
    return conversation;
  };

  useEffect(() => {
    const restoreConversation = async () => {
      try {
        const savedConversations = await refreshConversations();
        if (savedConversations[0]) await loadConversation(savedConversations[0].id);
        else await createConversation();
      } catch {
        setConnection('offline');
      } finally {
        setIsLoadingHistory(false);
      }
    };
    void restoreConversation();
  }, []);

  const sendMessage = async (event?: { preventDefault: () => void }) => {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || isStreaming || !conversationId) return;

    const userMessage: Message = { id: createMessageId(), role: 'user', content };
    const assistantId = createMessageId();
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      brain,
      isStreaming: true,
    };

    setDraft('');
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setIsStreaming(true);
    setConnection('working');

    try {
      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          brain,
          conversation_id: conversationId,
          messages: [{ role: 'user', content }],
        }),
      });
      if (!response.ok || !response.body) {
        throw new Error((await response.text()) || 'ATLAS could not start a response.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        pending += decoder.decode(value, { stream: true });
        const boundary = pending.lastIndexOf('\n\n');
        if (boundary === -1) continue;
        const completeFrames = pending.slice(0, boundary);
        pending = pending.slice(boundary + 2);

        for (const frame of streamFrames(completeFrames)) {
          if (frame.event === 'error') throw new Error(frame.data.detail || 'ATLAS stopped the response.');
          setMessages((current) =>
            current.map((message) => {
              if (message.id !== assistantId) return message;
              if (frame.event === 'token') {
                return { ...message, content: message.content + (frame.data.content ?? '') };
              }
              if (frame.event === 'tool_call') {
                const activity = toolActivityFromCalls(frame.data.tool_calls ?? []);
                return {
                  ...message,
                  toolActivity: [...(message.toolActivity ?? []), ...activity],
                };
              }
              if (frame.event === 'tool_result') {
                return {
                  ...message,
                  toolActivity: applyToolResults(message.toolActivity ?? [], frame.data.tool_results ?? []),
                };
              }
              if (frame.event === 'done') {
                return {
                  ...message,
                  brain: frame.data.brain ?? message.brain,
                  model: frame.data.model,
                  isStreaming: false,
                };
              }
              return message;
            }),
          );
        }
      }
      setConnection('ready');
      await refreshConversations();
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'ATLAS is unavailable.';
      setMessages(markConversationUnavailable(assistantId, detail));
      setConnection('offline');
    } finally {
      setIsStreaming(false);
    }
  };

  const startNewConversation = async () => {
    if (isStreaming || isLoadingHistory) return;
    try {
      await createConversation();
      setConnection('ready');
    } catch {
      setConnection('offline');
    }
  };

  const selectConversation = async (id: string) => {
    if (isStreaming || id === conversationId) return;
    try {
      await loadConversation(id);
      setConnection('ready');
    } catch {
      setConnection('offline');
    }
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  return (
    <main className="atlas-shell min-h-dvh overflow-hidden text-slate-100">
      <div className="atmosphere atmosphere-one" />
      <div className="atmosphere atmosphere-two" />
      <div className="matrix-grid" />

      <section className="relative mx-auto flex min-h-dvh max-w-[1760px] flex-col px-4 py-4 sm:px-6 sm:py-6">
        <header className="topbar glass-panel">
          <div className="flex min-w-0 items-center gap-3">
            <Button variant="ghost" size="icon" className="text-slate-400 hover:bg-white/5 hover:text-white" aria-label="Open navigation">
              <PanelLeft />
            </Button>
            <div className="atlas-mark" aria-hidden="true"><span /><span /><span /></div>
            <div className="min-w-0">
              <p className="text-[10px] font-semibold uppercase tracking-[0.26em] text-cyan-300/80">Adaptive, Thoughtful, Local Assistant System</p>
              <h1 className="truncate text-sm font-semibold tracking-[0.18em] text-white">A.T.L.A.S.</h1>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-white/8 bg-white/[0.025] px-3 py-1.5 text-xs text-slate-400 md:flex">
            <Radio className="size-3.5 text-cyan-300" /> Local mesh · Touwerij
          </div>
          <div className="flex items-center gap-2">
            <div className={`connection-chip connection-${connection}`}>
              <span className="status-orb" />
              <span className="hidden sm:inline">{connection === 'ready' ? 'Runtime ready' : connection === 'working' ? 'Thinking' : 'Runtime offline'}</span>
            </div>
            <Button variant="ghost" size="icon" className="text-slate-400 hover:bg-white/5 hover:text-white" aria-label="Settings"><Settings2 /></Button>
          </div>
        </header>

        <div className="grid flex-1 gap-4 pt-4 xl:grid-cols-[232px_minmax(0,1fr)_310px]">
          <aside className="sidebar-panel glass-panel hidden flex-col justify-between p-3 xl:flex">
            <div>
              <div className="mb-6 flex items-center justify-between px-2 pt-1"><p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Workspace</p><MoreHorizontal className="size-4 text-slate-600" /></div>
              <nav className="space-y-1" aria-label="ATLAS sections">
                <SidebarItem active icon={<MessageSquare />} label="Active conversation" />
                <SidebarItem icon={<History />} label="Memory" hint="Soon" />
                <SidebarItem icon={<Home />} label="Home state" hint="Soon" />
                <SidebarItem icon={<Activity />} label="Event feed" hint="Soon" />
              </nav>
              <div className="mt-6 border-t border-white/[0.06] pt-4">
                <p className="px-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Recent conversations</p>
                <div className="mt-2 space-y-1">
                  {conversations.slice(0, 6).map((conversation) => <button key={conversation.id} type="button" onClick={() => void selectConversation(conversation.id)} className={`conversation-item ${conversation.id === conversationId ? 'conversation-item-active' : ''}`} title={conversation.title}>{conversation.title}</button>)}
                </div>
              </div>
            </div>
            <div className="rounded-2xl border border-white/[0.07] bg-black/20 p-3">
              <div className="flex items-center gap-2 text-xs font-medium text-slate-200"><CircleDot className="size-3.5 text-cyan-300" />System posture</div>
              <p className="mt-2 text-xs leading-5 text-slate-500">Local first. No cloud routing configured.</p>
              <div className="mt-3 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.14em] text-cyan-200/70"><Check className="size-3" />Secure local mode</div>
            </div>
          </aside>

          <section className="chat-panel glass-panel relative flex min-h-[680px] flex-col overflow-hidden">
            <div className="chat-header">
              <div><p className="eyebrow">Private conversation</p><h2 className="mt-1 text-lg font-medium tracking-tight text-white sm:text-xl">What would you like to explore?</h2></div>
              <Button variant="ghost" size="sm" className="gap-1.5 text-slate-400 hover:bg-white/5 hover:text-white" onClick={() => void startNewConversation()} disabled={isStreaming || isLoadingHistory}><Plus className="size-3.5" />New thread</Button>
            </div>
            <div className="message-scrollbar flex-1 overflow-y-auto px-4 py-8 sm:px-9">
              <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
                <div className="system-line"><span /><p>Local history · Stored on this machine</p><span /></div>
                {messages.map((message) => <MessageBubble key={message.id} message={message} />)}
                <div ref={transcriptEnd} />
              </div>
            </div>
            <form onSubmit={sendMessage} className="composer-wrap">
              <div className="composer-glow" />
              <div className="composer relative">
                <Textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={handleComposerKeyDown} placeholder="Message ATLAS…" aria-label="Message ATLAS" className="min-h-[96px] resize-none border-0 bg-transparent px-4 py-4 pr-14 text-sm leading-6 text-white placeholder:text-slate-600 focus-visible:ring-0" disabled={isStreaming || isLoadingHistory || !conversationId} />
                <Button type="submit" size="icon-lg" className="send-button absolute bottom-3 right-3 rounded-xl" disabled={!draft.trim() || isStreaming || isLoadingHistory || !conversationId} aria-label="Send message"><ArrowUp className="size-4" /></Button>
                <div className="flex items-center gap-2 px-4 pb-3 text-[10px] font-medium text-slate-600"><Command className="size-3" />Enter to send <span className="mx-0.5 text-slate-800">·</span> Shift + Enter for a new line</div>
              </div>
            </form>
          </section>

          <aside className="brain-panel glass-panel order-first p-3 xl:order-none">
            <div className="flex items-center justify-between px-2 pb-3 pt-1"><div><p className="eyebrow">Neural routing</p><h2 className="mt-1 text-sm font-semibold text-white">Choose a brain</h2></div><ChevronDown className="size-4 text-slate-600" /></div>
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
              {(Object.keys(brains) as Brain[]).map((option) => {
                const selected = brain === option;
                const brainInfo = brains[option];
                return <button key={option} type="button" aria-pressed={selected} onClick={() => setBrain(option)} disabled={isStreaming} className={`brain-card brain-${brainInfo.accent} ${selected ? 'brain-selected' : ''}`}>
                  <span className="brain-icon">{option === 'fast' ? <Zap /> : <BrainCircuit />}</span>
                  <span className="min-w-0 flex-1 text-left"><span className="flex items-center justify-between gap-3"><span className="text-sm font-semibold text-white">{brainInfo.label}</span><span className="radio-dot" /></span><span className="mt-1 block text-[11px] font-medium text-slate-400">{brainInfo.model}</span><span className="mt-2 block text-xs leading-5 text-slate-500">{brainInfo.description}</span></span>
                </button>;
              })}
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2 border-t border-white/[0.06] pt-4 xl:grid-cols-1"><Metric icon={<Gauge />} label="Context" value="256K" /><Metric icon={<Cpu />} label="Execution" value="Local" /></div>
            <div className="mt-4 rounded-2xl border border-cyan-300/10 bg-cyan-300/[0.035] p-3.5"><div className="flex items-center gap-2 text-xs font-medium text-cyan-100"><Waves className="size-3.5 text-cyan-300" />Thoughtful by design</div><p className="mt-2 text-xs leading-5 text-slate-400">ATLAS records the selected brain with every response. Automatic escalation can be added later.</p></div>
          </aside>
        </div>
        <footer className="flex items-center justify-between gap-3 px-2 pt-3 text-[10px] font-medium text-slate-600"><span className="truncate normal-case tracking-normal">A.T.L.A.S. = Adaptive, Thoughtful, Local Assistant System</span><span className="flex shrink-0 items-center gap-1.5 uppercase tracking-[0.16em]"><Sparkles className="size-3 text-cyan-300/70" />Phase 2 interface</span></footer>
      </section>
    </main>
  );
}

function SidebarItem({ active = false, icon, label, hint }: { active?: boolean; icon: ReactNode; label: string; hint?: string }) {
  return <button type="button" className={`sidebar-item ${active ? 'sidebar-item-active' : ''}`}>{icon}<span className="flex-1 text-left">{label}</span>{hint && <span className="text-[9px] uppercase tracking-[0.1em] text-slate-600">{hint}</span>}</button>;
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return <div className="metric-card"><span>{icon}</span><div><p>{label}</p><strong>{value}</strong></div></div>;
}

function MessageBubble({ message }: { message: Message }) {
  const isAssistant = message.role === 'assistant';
  const selectedBrain = message.brain ? brains[message.brain] : undefined;
  return <article className={`message-row ${isAssistant ? 'message-assistant' : 'message-user'}`}>
    {isAssistant && <div className={`message-avatar ${message.brain === 'deep' ? 'avatar-deep' : ''}`} aria-hidden="true"><Sparkles className="size-3.5" /></div>}
    <div className="min-w-0">
      {isAssistant && <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">ATLAS{selectedBrain && <span className={`brain-badge brain-badge-${selectedBrain.accent}`}>{selectedBrain.label}</span>}</div>}
      <div className={`message-bubble ${isAssistant ? 'assistant-bubble' : 'user-bubble'}`}>
        {message.content
          ? isAssistant
            ? <AssistantMarkdown content={message.content} />
            : message.content
          : <span className="typing-dots"><i /><i /><i /></span>}
      </div>
      {isAssistant && message.toolActivity?.length ? <ToolActivityCard activity={message.toolActivity} isStreaming={message.isStreaming ?? false} /> : null}
      {isAssistant && (message.model || message.isStreaming) && <p className="mt-2 text-[10px] text-slate-600">{message.isStreaming ? 'Streaming local response…' : message.model}</p>}
    </div>
  </article>;
}

function AssistantMarkdown({ content }: { content: string }) {
  return <div className="markdown-content"><ReactMarkdown
      remarkPlugins={[remarkGfm]}
      skipHtml
      components={{
        a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a>,
      }}
    >{content}</ReactMarkdown></div>;
}

function ToolActivityCard({ activity, isStreaming }: { activity: ToolActivity[]; isStreaming: boolean }) {
  return <div className="tool-activity-card" aria-live="polite">
    {activity.map((tool) => {
      const isWeather = tool.name === 'get_current_weather';
      const isTime = tool.name === 'get_current_time';
      const label = isWeather ? 'Weather' : isTime ? 'Local time' : tool.name.replaceAll('_', ' ');
      const detail = tool.ok === false ? tool.content ?? 'Tool unavailable'
        : tool.ok === true ? toolResultSummary(tool, isWeather, isTime)
        : isWeather ? tool.location ? `Checking ${tool.location}` : 'Checking Waalwijk'
        : isTime ? 'Reading local time' : 'Using ATLAS tool';
      const Icon = isWeather ? CloudSun : isTime ? Clock3 : Wrench;
      return <div className="tool-activity-item" key={tool.id}>
        <span className={`tool-activity-icon ${isStreaming ? 'tool-activity-icon-active' : ''}`}><Icon className="size-3.5" /></span>
        <span className="min-w-0"><strong>{label}</strong><span>{isStreaming ? detail : 'Tool used for this response'}</span></span>
      </div>;
    })}
  </div>;
}

function toolResultSummary(tool: ToolActivity, isWeather: boolean, isTime: boolean): string {
  if (!tool.content) return 'Tool used for this response';
  try {
    const result = JSON.parse(tool.content) as Record<string, unknown>;
    if (isWeather && typeof result.location === 'string' && typeof result.temperature_c === 'number' && typeof result.condition === 'string') return `${result.location} · ${result.temperature_c}°C · ${result.condition}`;
    if (isTime && typeof result.local_time === 'string') return result.local_time;
  } catch { /* A tool may intentionally return plain text. */ }
  return 'Tool used for this response';
}
