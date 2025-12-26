import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";

const ModelComparison = ({ logistic, xgboost, lightgbm }) => {
  const data = [
    { name: "Logistic Regression", value: logistic, color: "#22c55e" },
    { name: "XGBoost", value: xgboost, color: "#22c55e" },
    { name: "LightGBM", value: lightgbm, color: "#f59e0b" },
  ];

  const renderCustomLabel = (props) => {
    const { x, y, width, value } = props;
    return (
      <text x={x + width / 2} y={y - 10} fill="#1e293b" textAnchor="middle" fontSize={12} fontWeight="bold">
        {value.toFixed(1)}%
      </text>
    );
  };

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <div>
          <h3>📊 Model Comparison</h3>
          <p style={{ fontSize: '12px', color: '#64748b', marginTop: '-12px' }}>
            Compare default probability across different ML models
          </p>
        </div>
        <div className="badge badge-final">⭐ LightGBM = Final Model</div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <XAxis dataKey="name" axisLine={false} tickLine={false} style={{ fontSize: '11px' }} />
          <YAxis domain={[0, 100]} hide />
          <Tooltip cursor={{ fill: 'transparent' }} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={50}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
            <LabelList dataKey="value" content={renderCustomLabel} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="card-footer">
        The LightGBM model is used as the primary decision engine due to its superior accuracy and 
        handling of complex feature interactions. Other models are shown for comparison.
      </div>
    </div>
  );
};

export default ModelComparison;