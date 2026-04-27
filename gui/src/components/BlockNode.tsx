import { Handle, Position, type NodeProps } from "reactflow";
import { useStore } from "../store";
import type { BlockNodeData } from "../types";

export function BlockNode({ id, data }: NodeProps<BlockNodeData>) {
  const blocks = useStore((s) => s.blocks);
  const videos = useStore((s) => s.videos);
  const setImpl = useStore((s) => s.setImpl);
  const setParam = useStore((s) => s.setParam);

  const impls = blocks[data.kind] ?? [];
  const showPath = data.kind === "input" || data.kind === "output";
  const path = (data.params.path as string | undefined) ?? "";

  // Input nodes get a dropdown sourced from data/. Output nodes keep the
  // plain text input (the user is naming an output, not picking an existing
  // file). Selecting from the dropdown sets path; the text input below it
  // works as a manual override / shows the current selection.
  const showVideoPicker = data.kind === "input";
  const pickerValue = videos.includes(path) ? path : "";

  return (
    <div className="block-node">
      <Handle type="target" position={Position.Left} />
      <div className="block-kind">{data.kind}</div>
      <select
        value={data.impl}
        onChange={(e) => setImpl(id, e.target.value)}
        className="nodrag"
      >
        {/* If the current impl isn't in the loaded list, show it anyway. */}
        {!impls.includes(data.impl) && (
          <option value={data.impl}>{data.impl}</option>
        )}
        {impls.map((impl) => (
          <option key={impl} value={impl}>
            {impl}
          </option>
        ))}
      </select>
      {showVideoPicker && (
        <select
          value={pickerValue}
          onChange={(e) => setParam(id, "path", e.target.value)}
          className="nodrag"
          title={
            videos.length === 0
              ? "No videos found in data/ — drop some clips there and refresh"
              : `${videos.length} video${videos.length === 1 ? "" : "s"} in data/`
          }
        >
          <option value="">
            {videos.length === 0
              ? "(no videos in data/)"
              : `— pick a clip (${videos.length}) —`}
          </option>
          {videos.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
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
