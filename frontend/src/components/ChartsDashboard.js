import {
  LineChart, Line,
  BarChart, Bar,
  PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";

const data = [
  { name: "Personal Loan", recovered: 65, cost: 40 },
  { name: "Home Loan", recovered: 80, cost: 25 },
  { name: "Auto Loan", recovered: 55, cost: 30 }
];

const riskData = [
  { name: "Low Risk", value: 55 },
  { name: "Medium Risk", value: 27 },
  { name: "High Risk", value: 18 }
];

const heatData = [
  { segment: "Low Income", risk: 0.2 },
  { segment: "Medium Income", risk: 0.5 },
  { segment: "High Income", risk: 0.8 }
];

const waterfallData = [
  { name: "Total Amount", value: 120 },
  { name: "Recovered", value: -78 },
  { name: "Pending", value: 42 }
];

const COLORS = ["#2ecc71", "#f1c40f", "#e74c3c"];

export default function ChartsDashboard() {
  return (
    <div className="chart-grid">

      {/* Line Chart */}
      <div className="chart-box">
        <h4>Recovery % by Loan Type</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Line dataKey="recovered" stroke="#3498db" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Bar Chart */}
      <div className="chart-box">
        <h4>Total Cost by Loan Type</h4>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="cost" fill="#9b59b6" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Pie Chart */}
      <div className="chart-box">
        <h4>Risk Distribution</h4>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={riskData} dataKey="value" outerRadius={80}>
              {riskData.map((_, i) => (
                <Cell key={i} fill={COLORS[i]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Heatmap */}
      <div className="chart-box">
        <h4>Risk Heatmap (Customer Segments)</h4>
        {heatData.map((d, i) => (
          <div
            key={i}
            style={{
              padding: "10px",
              margin: "6px",
              borderRadius: "6px",
              background:
                d.risk > 0.6 ? "#e74c3c" :
                d.risk > 0.3 ? "#f1c40f" :
                "#2ecc71",
              color: "white"
            }}
          >
            {d.segment} → Risk: {Math.round(d.risk * 100)}%
          </div>
        ))}
      </div>

      {/* Waterfall (simplified) */}
      <div className="chart-box">
        <h4>Recovery Waterfall</h4>
        {waterfallData.map((d, i) => (
          <p key={i}>
            <strong>{d.name}:</strong> {d.value}
          </p>
        ))}
      </div>

    </div>
  );
}
