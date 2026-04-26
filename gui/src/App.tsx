import { useEffect } from "react";
import { Canvas } from "./components/Canvas";
import { Palette } from "./components/Palette";
import { RunPanel } from "./components/RunPanel";
import { useStore } from "./store";

export default function App() {
  const loadBlocks = useStore((s) => s.loadBlocks);

  useEffect(() => {
    loadBlocks().catch((err) => console.error("loadBlocks failed", err));
  }, [loadBlocks]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>ModernCV</h1>
      </header>
      <div className="app-body">
        <Palette />
        <main className="canvas-pane">
          <Canvas />
        </main>
      </div>
      <RunPanel />
    </div>
  );
}
