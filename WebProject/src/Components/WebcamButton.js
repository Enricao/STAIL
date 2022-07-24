import React from 'react';
import ReactDOM from 'react-dom';
import Webcam from 'react-webcam';
import './WebcamButton.css'

const appRoot = document.getElementById('root');
const videoConstraints = {
  width: 640,
  height: 640,
  facingMode: "user"
};
const maxVideoSize = 640

class WebcamComponent extends React.Component {
  webcam = React.createRef();
  webcamContainer = React.createRef();

  constructor(props) {
    super(props);
    this.state = {
      mountedWebcam: false,
      mountedInterface: false
    };
  }

  mountWebcam() {
    const size = Math.min(window.innerWidth, window.innerHeight, maxVideoSize + 32) - 32;
    videoConstraints.width = videoConstraints.height = size;
    this.setState({mountedWebcam: true});
  }

  dismountWebcam() {
    this.setState({
      mountedWebcam: false,
      mountedInterface: false
    });
    this.webcamContainer.current?.classList.remove('scard')
  }

  webcamDidMount() {
    new ResizeObserver(() => this.containerSizeChange()).observe(this.webcamContainer.current);
  }

  takePicture() {
    let image = this.webcam.current.getScreenshot();
    this.props.setImage(image);
    this.dismountWebcam();
  }

  containerSizeChange() {
    let el = this.webcamContainer.current
    if (el) {
      let rect = this.webcamContainer.current.getBoundingClientRect();
      if (videoConstraints.width == rect.width) {
        el.classList.add('scard')
        this.setState({
          mountedInterface: true
        })
      }
    }
  }

  render() {
    return (
      <>
        <button style={this.props.style} className={this.props.className} type="button" onClick={() => this.mountWebcam()}>
          {this.props.children}
        </button>
        {this.state.mountedWebcam ? ReactDOM.createPortal(
          <div className="container-container position-fixed fixed-top h-100 d-flex justify-content-center align-items-center"
            onClick={() => this.dismountWebcam()}>
            <div ref={this.webcamContainer} className="webcam-container position-relative"
              style={{width: videoConstraints.width, height: videoConstraints.height}}>
              <div className="webcam-error"></div>
              <Webcam className="webcam"
                ref={this.webcam}
                screenshotFormat="image/jpeg"
                mirrored
                videoConstraints={videoConstraints}
                onUserMedia={() => this.webcamDidMount()}
                onUserMediaError={() => this.dismountWebcam()} />
              {this.state.mountedInterface ? <>
                <button type="button" className="btn-close" aria-label="Close" onClick={() => this.dismountWebcam()}></button>
                <button type="button" className="btn-picture" onClick={() => this.takePicture()}></button>
                <div className="webcam-cover" ref={(el) => (el && setTimeout(() => el.classList.add('hidden'), 1000))}>
                  Line up your eyes and take a picture!
                </div>
                <div className="left-eye hidden" ref={(el) => (el && setTimeout(() => el.classList.remove('hidden'), 1000))}></div>
                <div className="right-eye hidden" ref={(el) => (el && setTimeout(() => el.classList.remove('hidden'), 1000))}></div>
              </> : null}
            </div>
          </div>,
          appRoot
        ) : null}
      </>
    )
  }
}

export default WebcamComponent
