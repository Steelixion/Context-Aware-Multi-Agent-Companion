import React, { useLayoutEffect, useRef } from "react";
import { Card, Chip } from "./ui.jsx";

export default function ChatPane({ chat, speaking, llmOnline }) {
  const box = useRef(null);
  const pinned = useRef(true);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el || !pinned.current) return;
    el.scrollTop = el.scrollHeight;
  }, [chat]);

  const rememberPinned = () => {
    const el = box.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  return (
    <Card title="Conversation history" className="flex h-full min-h-[24rem] flex-col"
      right={<div className="flex gap-2">
        <Chip tone={llmOnline ? "green" : "amber"}>{llmOnline ? "LLM online" : "LLM offline"}</Chip>
        {speaking && <Chip tone={speaking.priority === 0 ? "red" : "green"}>{speaking.priority === 0 ? "WARNING" : "speaking"}</Chip>}
      </div>}>
      <div ref={box} onScroll={rememberPinned}
        className="flex-1 space-y-2 overflow-y-auto overscroll-contain pr-1">
        {chat.length === 0 && <p className="text-sm text-slate-500">No conversation history yet — spoken turns appear here as they happen.</p>}
        {chat.map((m) => (
          <div key={m.id} className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${m.role === "user" ? "ml-auto bg-sky-600/30" : "bg-slate-800"}`}>
            {m.text || <em className="text-slate-500">(interrupted)</em>}
            <span className="ml-2 text-xs text-slate-400">
              {(() => {
                const date = new Date(m.ts * 1000);
                const opts = { hour: "2-digit", minute: "2-digit", second: "2-digit" };
                const timeStr = date.toLocaleTimeString(undefined, { ...opts, fractionalSecondDigits: 3 });
                return timeStr;
              })()}
            </span>
            {m.interrupted && m.text && <em className="ml-1 text-xs text-slate-500">(interrupted)</em>}
            {m.emotion && m.role === "user" && <span className="ml-2 text-[10px] uppercase text-slate-400">{m.emotion}</span>}
          </div>
        ))}
      </div>
    </Card>
  );
}
