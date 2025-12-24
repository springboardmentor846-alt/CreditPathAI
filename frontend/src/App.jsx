import LoanForm from "./components/LoanForm";
import ResultCard from "./components/ResultCard";
import { useState } from "react";
import "./App.css";
import ModelComparison from "./components/ModelComparison";
import ModelPerformance from "./components/ModelPerformance";

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Loan Default Risk Prediction</h1>
        <p>AI-powered credit risk assessment dashboard</p>
      </header>

      <LoanForm onResult={setResult} />
      <ResultCard result={result} />
      <ModelComparison />
      <ModelPerformance />

      <footer className="app-footer">
        © 2025 Loan Risk Analytics • ML Powered
      </footer>
    </div>
  );
}

export default App;
