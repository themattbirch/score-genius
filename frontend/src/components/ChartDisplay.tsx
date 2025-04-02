// frontend/src/components/ChartDisplay.tsx
import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface ChartDisplayProps {
  data: { name: string; value: number }[];
}

const ChartDisplay: React.FC<ChartDisplayProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="value" stroke="#1e90ff" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ChartDisplay;
