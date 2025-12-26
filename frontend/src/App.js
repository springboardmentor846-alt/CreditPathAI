import KPICards from "./components/KPICards";
import ChartsDashboard from "./components/ChartsDashboard";
import RecommendationPanel from "./components/RecommendationPanel";

function App() {
  return (
    <div className="container">
      <h1 className="title">Smart Loan Recovery System</h1>

      <KPICards />
      <ChartsDashboard />
      <RecommendationPanel />
    </div>
  );
}

export default App;
