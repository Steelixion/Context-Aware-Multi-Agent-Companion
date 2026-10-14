import React, { useEffect, useRef, useState } from "react";
import { Card, Chip } from "./ui.jsx";

const EMOTIONS = ["neutral", "happy", "sad", "stressed", "angry"];

export default function ChatPane({ chat, sendChat, speaking, info }) {
  const [text, setText] = useState("");
  const [emotion, setEmotion] = useState("neutral");
  const end = useRef(null);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  const submit = (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    sendChat(text, info?.mic ? undefined : emotion);
    setText("");
  };

  return (
    <Card title="Companion" className="flex h-full min-h-[24rem] flex-col"
      right={<div className="flex gap-2">
        <Chip tone={info?.llm_online ? "green" : "amber"}>{info?.llm_online ? "LLM online" : "LLM offline"}</Chip>
        {speaking && <Chip tone={speaking.priority === 0 ? "red" : "green"}>{speaking.priority === 0 ? "WARNING" : "speaking"}</Chip>}
      </div>}>
      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {chat.length === 0 && <p className="text-sm text-slate-500">Say something or type below. Facts you mention are remembered across sessions.</p>}
        {chat.map((m) => (
          <div key={m.id} className={`max-w-[85%] rounded-2xl px-3 py-2 text-sm ${m.role === "user" ? "ml-auto bg-sky-600/30" : "bg-slate-800"}`}>
            {m.text || <em className="text-slate-500">(interrupted)</em>}
            {m.interrupted && m.text && <em className="ml-1 text-xs text-slate-500">(interrupted)</em>}
            {m.emotion && m.role === "user" && <span className="ml-2 text-[10px] uppercase text-slate-400">{m.emotion}</span>}
          </div>
        ))}
        <div ref={end} />
      </div>
      <form onSubmit={submit} className="mt-3 flex gap-2">
        {!info?.mic && (
          <select value={emotion} onChange={(e) => setEmotion(e.target.value)}
            title="Simulated speech emotion (no microphone enabled)"
            className="rounded-lg border border-slate-700 bg-slate-800 px-2 text-sm">
            {EMOTIONS.map((e) => <option key={e}>{e}</option>)}
          </select>
        )}
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a message…"
          className="min-w-0 flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm outline-none focus:border-sky-500" />
        <button className="rounded-lg bg-sky-600 px-4 text-sm font-medium hover:bg-sky-500">Send</button>
      </form>
    </Card>
  );
}
