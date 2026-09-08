import React from "react";
import { Alert, Button, Card } from "antd";

export type BrowserVideoMetadata = { width: number; height: number; duration: number };

export function readVideoMetadata(video: HTMLVideoElement): BrowserVideoMetadata {
  return {
    width: video.videoWidth,
    height: video.videoHeight,
    duration: video.duration,
  };
}

export function mediaErrorText(video: HTMLVideoElement): string {
  const code = video.error?.code;
  const detail = video.error?.message;
  return `视频 ${video.dataset.camera || "unknown"} 播放失败${code ? `（错误码 ${code}）` : ""}${detail ? `：${detail}` : ""}`;
}

type VideoPreviewErrorBoundaryProps = {
  children: React.ReactNode;
  onRetry: () => void;
};

type VideoPreviewErrorBoundaryState = {
  error: Error | null;
};

export class VideoPreviewErrorBoundary extends React.Component<
  VideoPreviewErrorBoundaryProps,
  VideoPreviewErrorBoundaryState
> {
  state: VideoPreviewErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): VideoPreviewErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // Keep the error observable for diagnostics while showing a local fallback.
    console.error("视频预览区域渲染失败", error, info.componentStack);
  }

  private retry = (): void => {
    this.setState({ error: null });
    this.props.onRetry();
  };

  render(): React.ReactNode {
    if (!this.state.error) return this.props.children;
    return <Card title="视频预览暂时不可用" className="section-card video-preview-fallback">
      <Alert
        type="error"
        showIcon
        title="视频预览区域发生异常"
        description={this.state.error.message || "请重新加载当前 Episode 预览。"}
        action={<Button onClick={this.retry}>重新加载预览</Button>}
      />
    </Card>;
  }
}

type CameraVideoProps = {
  camera: string;
  src: string;
  target: number;
  retryToken: number;
  error?: string;
  onClick: () => void;
  onLoadedMetadata: (
    camera: string,
    video: HTMLVideoElement,
    metadata: BrowserVideoMetadata,
  ) => void;
  onWaiting: () => void;
  onStalled: () => void;
  onCanPlay: () => void;
  onError: (camera: string, video: HTMLVideoElement, message: string) => void;
  onRetry: () => void;
  setRef: (element: HTMLVideoElement | null) => void;
};

export function CameraVideo({
  camera,
  src,
  target,
  retryToken,
  error,
  onClick,
  onLoadedMetadata,
  onWaiting,
  onStalled,
  onCanPlay,
  onError,
  onRetry,
  setRef,
}: CameraVideoProps): React.ReactElement {
  if (error) {
    return <div className="camera-video-error" role="alert">
      <Alert
        type="error"
        showIcon
        title={`${camera} 视频加载失败`}
        description={error}
        action={<Button size="small" onClick={onRetry}>重试该摄像头</Button>}
      />
    </div>;
  }

  return <video
    key={`${camera}:${src}:${retryToken}`}
    className="episode-video"
    playsInline
    muted
    preload="metadata"
    src={src}
    data-camera={camera}
    aria-label={`${camera} 视频`}
    onClick={onClick}
    ref={setRef}
    onLoadedMetadata={(event) => {
      // Capture the element and all metadata before any state update can be
      // deferred. No React event is passed beyond this synchronous callback.
      const video = event.currentTarget;
      onLoadedMetadata(camera, video, readVideoMetadata(video));
      if (Math.abs(video.currentTime - target) > 0.1) video.currentTime = target;
    }}
    onWaiting={onWaiting}
    onStalled={onStalled}
    onCanPlay={onCanPlay}
    onError={(event) => {
      const video = event.currentTarget;
      onError(camera, video, mediaErrorText(video));
    }}
  />;
}
