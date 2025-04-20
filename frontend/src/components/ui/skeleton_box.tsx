const SkeletonBox: React.FC<{ className?: string }> = ({ className }) => (
  <div className={`animate-pulse rounded bg-slate-700/50 ${className}`} />
);
export default SkeletonBox;
