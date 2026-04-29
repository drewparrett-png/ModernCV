import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchTrainingCurve, type TrainingCurve } from "../api";

/**
 * Two small line charts: training loss (descending = good) and
 * validation mAP (ascending = good). Reads Ultralytics' results.csv
 * via the /training_curve endpoint.
 *
 * Lazy-loads only when mounted, so the data fetch only happens when
 * the user opens a Student that has a curve to show.
 */
export function TrainingCurves({
  projectId,
  studentId,
  status,
}: {
  projectId: string;
  studentId: string;
  status: string;
}) {
  const [curve, setCurve] = useState<TrainingCurve | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchTrainingCurve(projectId, studentId)
      .then((c) => {
        if (cancelled) return;
        setCurve(c);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [projectId, studentId, status]);

  if (loading) {
    return <div className="curves-empty muted">Loading curves…</div>;
  }
  if (error) {
    return <div className="curves-empty muted">Couldn't load curves: {error}</div>;
  }
  if (!curve || curve.epochs.length === 0) {
    return (
      <div className="curves-empty muted">
        Curves will appear after the first epoch.
      </div>
    );
  }

  const lossData = curve.epochs.map((e, i) => ({
    epoch: e,
    train_loss: curve.train_loss[i],
  }));
  const mapData = curve.epochs.map((e, i) => ({
    epoch: e,
    map50: curve.val_map50[i],
    map50_95: curve.val_map50_95[i],
  }));

  return (
    <div className="training-curves">
      <div className="training-curve">
        <span className="training-curve-title">Train loss</span>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={lossData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <CartesianGrid stroke="#f3f4f6" strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#9ca3af"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#9ca3af"
              width={36}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, padding: "4px 8px" }}
              labelStyle={{ fontSize: 11 }}
            />
            <Line
              type="monotone"
              dataKey="train_loss"
              stroke="#dc2626"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="training-curve">
        <span className="training-curve-title">Validation mAP</span>
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={mapData} margin={{ top: 4, right: 8, left: -16, bottom: 4 }}>
            <CartesianGrid stroke="#f3f4f6" strokeDasharray="3 3" />
            <XAxis
              dataKey="epoch"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#9ca3af"
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#9ca3af"
              width={36}
            />
            <Tooltip
              contentStyle={{ fontSize: 11, padding: "4px 8px" }}
              labelStyle={{ fontSize: 11 }}
            />
            <Line
              type="monotone"
              dataKey="map50"
              name="mAP@0.5"
              stroke="#16a34a"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="map50_95"
              name="mAP@0.5:0.95"
              stroke="#2563eb"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
