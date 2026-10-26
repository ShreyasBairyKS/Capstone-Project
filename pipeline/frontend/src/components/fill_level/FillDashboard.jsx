import React from 'react';
import FillKpiBar from './FillKpiBar';
import FillSingleFeed from './FillSingleFeed';
import FillConveyorFilmstrip from './FillConveyorFilmstrip';
import FillAnalyticsWidget from './FillAnalyticsWidget';

export default function FillDashboard({
  stats,
  latestInspection,
  inspectionHistory,
  defects,
  onSelectBottle
}) {
  return (
    <div className="flex flex-col gap-3.5 w-full">
      {/* 1. Station 2 KPI Banner Bar */}
      <FillKpiBar 
        stats={stats} 
        latestLatency={latestInspection?.total_latency_ms} 
      />

      {/* 2. Single Camera Profile Viewport with Meniscus Line & Measurement Ruler */}
      <FillSingleFeed 
        latestInspection={latestInspection}
        onInspectBottle={onSelectBottle}
      />

      {/* 3. Real-Time Conveyor Filmstrip Queue */}
      <FillConveyorFilmstrip 
        inspectionHistory={inspectionHistory}
        onSelectBottle={onSelectBottle}
      />

      {/* 4. Fill Volume Distribution & Yield Analytics */}
      <FillAnalyticsWidget 
        stats={stats}
        defects={defects}
      />
    </div>
  );
}
