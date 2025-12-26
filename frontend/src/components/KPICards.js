export default function KPICards() {
  const kpis = [
    { label: "Total Cases", value: "50,000" },
    { label: "Total Amount", value: "₹120 Cr" },
    { label: "Recovered Amount", value: "₹78 Cr" },
    { label: "Recovery %", value: "65%" },
    { label: "High Risk %", value: "18%" }
  ];

  return (
    <div className="kpi-grid">
      {kpis.map((kpi, i) => (
        <div className="kpi-card" key={i}>
          <h3>{kpi.value}</h3>
          <p>{kpi.label}</p>
        </div>
      ))}
    </div>
  );
}
