// dashboard/src/components/Spinner.jsx
import React from "react";

export default function Spinner({ size = 36 }) {
  const style = { width: size, height: size };
  return (
    <div style={{ display: "inline-block", verticalAlign: "middle" }}>
      <div className="spinner-border text-light" role="status" style={style}>
        <span className="visually-hidden">Loading...</span>
      </div>
    </div>
  );
}