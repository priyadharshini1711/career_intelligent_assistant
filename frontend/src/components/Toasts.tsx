export interface Toast {
  id: string
  tone: 'error' | 'info'
  title: string
  detail?: string
}

export default function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: string) => void }) {
  if (!toasts.length) return null

  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div className={`toast ${toast.tone}`} key={toast.id}>
          <span className="toast-icon">{toast.tone === 'error' ? '!' : 'i'}</span>
          <div className="toast-body">
            <div className="toast-title">{toast.title}</div>
            {toast.detail && <div className="toast-detail">{toast.detail}</div>}
          </div>
          <button className="btn-subtle" onClick={() => onDismiss(toast.id)} aria-label="Dismiss">
            ×
          </button>
        </div>
      ))}
    </div>
  )
}
