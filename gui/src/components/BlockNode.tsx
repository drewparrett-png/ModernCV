import { Handle, Position, type NodeProps } from "reactflow";
import { useStore } from "../store";
import type { BlockNodeData } from "../types";
import { blockMeta, implMeta } from "../blockMeta";
import { VideoTreePicker } from "./VideoTreePicker";

export function BlockNode({ id, data }: NodeProps<BlockNodeData>) {
  const blocks = useStore((s) => s.blocks);
  const videos = useStore((s) => s.videos);
  const setImpl = useStore((s) => s.setImpl);
  const setParam = useStore((s) => s.setParam);

  const kindMeta = blockMeta(data.kind);
  const currentImpl = implMeta(data.kind, data.impl);

  const impls = blocks[data.kind] ?? [];
  const showPath = data.kind === "input" || data.kind === "output";
  const path = (data.params.path as string | undefined) ?? "";

  // Input nodes use the folder-tree picker sourced from data/. Output nodes
  // keep the plain text input (the user is naming an output, not picking
  // an existing file). The text input below the picker on input nodes
  // works as a manual override / shows the current selection.
  const showVideoPicker = data.kind === "input";

  return (
    <div className="block-node">
      <Handle type="target" position={Position.Left} />
      <div className="block-kind" title={kindMeta.description}>
        {kindMeta.label}
      </div>
      <select
        value={data.impl}
        onChange={(e) => setImpl(id, e.target.value)}
        className="nodrag"
        title={currentImpl.description}
      >
        {/* If the current impl isn't in the loaded list, show it anyway. */}
        {!impls.includes(data.impl) && (
          <option value={data.impl}>{currentImpl.label}</option>
        )}
        {impls.map((implId) => {
          const info = implMeta(data.kind, implId);
          return (
            <option key={implId} value={implId} title={info.description}>
              {info.label}
            </option>
          );
        })}
      </select>
      {currentImpl.description && (
        <div className="block-impl-help">{currentImpl.description}</div>
      )}
      {showVideoPicker && (
        <div className="nodrag">
          <VideoTreePicker
            videos={videos}
            selected={path}
            onSelect={(v) => setParam(id, "path", v)}
          />
        </div>
      )}
      {showPath && (
        <input
          type="text"
          value={path}
          placeholder="path"
          onChange={(e) => setParam(id, "path", e.target.value)}
          className="nodrag"
        />
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
