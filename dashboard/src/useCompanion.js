import { useCallback, useEffect, useRef, useState } from "react";

const HISTORY = 300; // ~30 s at 10 Hz

export function useCompanion() {
  const [connected, setConnected] = useState(false);
  const [tel, setTel] = useState(null);
  const [series, setSeries] = useState([]);
  const [frame, setFrame] = useState(null);
  const [alert, setAlert] = useState(null);       // {text, severity, until}
  const [alerts, setAlerts] = useState([]);
  const [chat, setChat] = useState([]);
  const [speaking, setSpeaking] = useState(null);
  const [sync, setSync] = useState(null);
  const [info, setInfo] = useState(null);
  const [memTick, setMemTick] = useState(0);
  const ws = useRef(null);
  const buf = useRef([]);

  useEffect(() => {
    let closed = false, timer;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/ws`);
      ws.current = sock;
      sock.onopen = () => setConnected(true);
      sock.onclose = () => { setConnected(false); if (!closed) timer = setTimeout(connect, 1500); };
      sock.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        const { type, data } = msg;
        if (type === "snapshot") { setChat(msg.chat || []); setSync(msg.sync); }
        else if (type === "telemetry") {
          setTel(data);
          buf.current = [...buf.current.slice(-(HISTORY - 1)), { t: data.ts, ear: data.ear, fatigue: data.fatigue_score, mar: data.mar }];
        }
        else if (type === "frame") setFrame(data);
        else if (type === "alert") {
          setAlert({ ...data, until: Date.now() + 8000 });
          setAlerts((a) => [data, ...a].slice(0, 20));
        }
        else if (type === "chat") setChat((c) => {
          const i = c.findIndex((m) => m.id === data.id);
          if (i >= 0) { const n = [...c]; n[i] = { ...n[i], ...data }; return n; }
          return [...c, data].slice(-60);
        });
        else if (type === "audio") setSpeaking(data.speaking ? data : null);
        else if (type === "sync") setSync(data);
        else if (type === "memory") setMemTick((n) => n + 1);
      };
    };
    connect();
    const flush = setInterval(() => setSeries(buf.current), 250);
    const expire = setInterval(() => setAlert((a) => (a && a.until < Date.now() ? null : a)), 500);
    return () => { closed = true; clearTimeout(timer); clearInterval(flush); clearInterval(expire); ws.current?.close(); };
  }, []);

  useEffect(() => { fetch("/api/session").then((r) => r.json()).then(setInfo).catch(() => {}); }, [connected]);

  const send = useCallback((obj) => ws.current?.readyState === 1 && ws.current.send(JSON.stringify(obj)), []);
  return {
    connected, tel, series, frame, alert, alerts, chat, speaking, sync, info, memTick,
    sendChat: (text, emotion) => send({ type: "chat", text, emotion }),
    setSim: (mode) => send({ type: "sim", mode }),
  };
}
