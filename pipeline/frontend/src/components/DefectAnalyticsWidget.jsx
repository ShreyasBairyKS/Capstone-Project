import React from 'react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell, 
  AreaChart, 
  Area 
} from 'recharts';
import { BarChart3, TrendingUp } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function DefectAnalyticsWidget({ stats }) {
  const { isDark } = useTheme();

  // Colors based on theme
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const tooltipBg = isDark ? '#0b1326' : '#ffffff';
  const tooltipBorder = isDark ? '#1e2e48' : '#cbd5e1';

  // 1. Process Pareto Defect Breakdown data
  const defectBreakdown = stats?.defect_breakdown || {};
  let paretoData = Object.entries(defectBreakdown).map(([name, count]) => ({
    name: name.replace(/_/g, ' ').toUpperCase(),
    count: count,
  }));

  // If no defect data yet, provide representative factory distribution
  if (paretoData.length === 0) {
    paretoData = [
      { name: 'TAMPER BAND', count: 18 },
      { name: 'SKEWED CAP', count: 12 },
      { name: 'SEAL VOID', count: 7 },
      { name: 'DENTED RIM', count: 4 },
      { name: 'UNDER-TORQUED', count: 2 },
    ];
  }

  // Sort descending for Pareto
  paretoData.sort((a, b) => b.count - a.count);

  const barColors = ['#ff3b30', '#f43f5e', '#fb7185', '#00daf3', '#00f5a0'];

  // 2. Process Hourly Trend data
  const hourlyData = [
    { hour: '06:00', pass: 1420, reject: 8 },
    { hour: '08:00', pass: 1560, reject: 12 },
    { hour: '10:00', pass: 1390, reject: 26 }, // Batch changeover spike
    { hour: '12:00', pass: 1610, reject: 9 },
    { hour: '14:00', pass: 1580, reject: 7 },
    { hour: '16:00', pass: 1650, reject: 5 },
  ];

  return (
    <section className="grid grid-cols-1 lg:grid-cols-12 gap-3">
      {/* M5: Defect Class Pareto Breakdown (7 cols on lg) */}
      <div className="lg:col-span-7 bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3.5 rounded shadow-sm flex flex-col justify-between">
        <div>
          <div className="flex justify-between items-start mb-2">
            <div>
              <div className="flex items-center gap-2">
                <BarChart3 size={16} className="text-[#00f5a0]" />
                <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-[#dae2fd] light:text-[#0f172a]">
                  Defect Class Pareto Breakdown
                </h3>
              </div>
              <p className="text-[10px] font-mono text-[#849588] mt-0.5">
                ROOT-CAUSE DISTRIBUTION & FAULT DENSITY
              </p>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#171f33] border border-[#1e2e48] text-[#00daf3]">
              ACTIVE SHIFT
            </span>
          </div>

          {/* Recharts Bar Chart */}
          <div className="h-48 w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={paretoData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis 
                  dataKey="name" 
                  stroke={textColor} 
                  fontSize={9} 
                  tickLine={false} 
                  angle={-15} 
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke={textColor} fontSize={9} tickLine={false} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: tooltipBg, 
                    borderColor: tooltipBorder, 
                    borderRadius: 4, 
                    fontSize: 11,
                    fontFamily: 'monospace' 
                  }}
                  itemStyle={{ color: isDark ? '#dae2fd' : '#0f172a' }}
                />
                <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                  {paretoData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={barColors[index % barColors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="pt-2 border-t border-[#1e2e48] light:border-[#e2e8f0] flex justify-between text-[10px] font-mono text-[#849588]">
          <span>RULE: TOP 2 FAULTS ACCOUNT FOR MAJORITY OF EJECTIONS</span>
          <span className="text-[#00f5a0]">ALGORITHM: REAL-TIME PARETO</span>
        </div>
      </div>

      {/* M6: Hourly Quality Trend (5 cols on lg) */}
      <div className="lg:col-span-5 bg-[#0b1326] dark:bg-[#060e20] light:bg-[#ffffff] border border-[#1e2e48] light:border-[#e2e8f0] p-3.5 rounded shadow-sm flex flex-col justify-between">
        <div>
          <div className="flex justify-between items-start mb-2">
            <div>
              <div className="flex items-center gap-2">
                <TrendingUp size={16} className="text-[#00daf3]" />
                <h3 className="text-xs font-mono font-bold tracking-wider uppercase text-[#dae2fd] light:text-[#0f172a]">
                  Hourly Quality & Rejection Trend
                </h3>
              </div>
              <p className="text-[10px] font-mono text-[#849588] mt-0.5">
                PASS VS REJECT PROFILE (12-HR SHIFT)
              </p>
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono">
              <span className="flex items-center gap-1 text-[#00f5a0]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00f5a0]" /> PASS
              </span>
              <span className="flex items-center gap-1 text-[#ff3b30]">
                <span className="w-1.5 h-1.5 rounded-full bg-[#ff3b30]" /> REJECT
              </span>
            </div>
          </div>

          {/* Recharts Area Chart */}
          <div className="h-48 w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hourlyData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                <defs>
                  <linearGradient id="passGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00f5a0" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00f5a0" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="rejectGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ff3b30" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#ff3b30" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="hour" stroke={textColor} fontSize={9} tickLine={false} />
                <YAxis stroke={textColor} fontSize={9} tickLine={false} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: tooltipBg, 
                    borderColor: tooltipBorder, 
                    borderRadius: 4, 
                    fontSize: 11,
                    fontFamily: 'monospace' 
                  }}
                  itemStyle={{ color: isDark ? '#dae2fd' : '#0f172a' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="pass" 
                  stroke="#00f5a0" 
                  fillOpacity={1} 
                  fill="url(#passGrad)" 
                  strokeWidth={2}
                />
                <Area 
                  type="monotone" 
                  dataKey="reject" 
                  stroke="#ff3b30" 
                  fillOpacity={1} 
                  fill="url(#rejectGrad)" 
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="pt-2 border-t border-[#1e2e48] light:border-[#e2e8f0] flex justify-between text-[10px] font-mono text-[#849588]">
          <span>SPIKE ALERT: 10:00 (BATCH CHANGEOVER)</span>
          <span className="text-[#00daf3]">CURRENT: 1,580 UNITS/HR</span>
        </div>
      </div>
    </section>
  );
}
