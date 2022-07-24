import LoadingButton from '../Components/LoadingButton'
import WebcamButton from '../Components/WebcamButton'
import './UploadImage.css'

const React = require('react')

function urltoFile(url, filename, mimeType){
    return (fetch(url)
        .then(function(res){return res.arrayBuffer();})
        .then(function(buf){return new File([buf], filename,{type:mimeType});})
    );
}

class UploadImage extends React.Component {
    constructor() {
        super();
        this.state = {
            file: null,
            name: "Choose Image"
        }
        this.handler = this.handler.bind(this)
    }

    handler(event) {
        let file = event.target.files[0];
        console.log(file);
        if (file) {
            this.name = file.name;
            this.setState({
                fileURL: URL.createObjectURL(file),
                name: file.name,
                file: file
            })
        }
    }

    setCameraImage(src) {
        urltoFile(src, 'image.jpg', 'image/jpeg').then((file) => {
            this.setState({
                fileURL: src,
                name: "Camera Image",
                file: file
            })
        })
    }

    render() {
        return (
            <div className="container">

                <header className="text-white text-center mb-5">
                    <h1 className="text-white font-weight-normal home-2-title display-4 mb-0 text-center">Upload an
                        Image</h1>
                    <p className="lead mb-4">So you can get a refreshing recommendation</p>
                </header>

                <div className="home-center mt-5">
                    <div className="home-desc-center">
                        <div className="container">
                            <div className="align-items-center row">
                                <div className="col-lg-12">
                                    <div className="mt-20">
                                        <div className="row py-4">
                                            <div className="col-lg-6 mx-auto">
                                                <div className="input-group mb-2 px-2 py-2 rounded-pill bg-white shadow-sm">  {/*  */}
                                                    <div className="input-group-prepend">
                                                        <WebcamButton className="webcam-button btn btn-outline-secondary rounded-pill" setImage={(src) => this.setCameraImage(src)}>
                                                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-camera" viewBox="0 0 16 16">
                                                                <path d="M15 12a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h1.172a3 3 0 0 0 2.12-.879l.83-.828A1 1 0 0 1 6.827 3h2.344a1 1 0 0 1 .707.293l.828.828A3 3 0 0 0 12.828 5H14a1 1 0 0 1 1 1v6zM2 4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V6a2 2 0 0 0-2-2h-1.172a2 2 0 0 1-1.414-.586l-.828-.828A2 2 0 0 0 9.172 2H6.828a2 2 0 0 0-1.414.586l-.828.828A2 2 0 0 1 3.172 4H2z"/>
                                                                <path d="M8 11a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5zm0 1a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM3 6.5a.5.5 0 1 1-1 0 .5.5 0 0 1 1 0z"/>
                                                            </svg>
                                                        </WebcamButton>
                                                    </div>
                                                    <input id="upload" type="file" onChange={this.handler}
                                                        className="form-control border-0"/>
                                                    <label id="upload-label" htmlFor="upload"
                                                        className="font-weight-light text-muted">{this.state.name}</label>
                                                    <div className="input-group-append">
                                                        <label htmlFor="upload" className="btn btn-outline-secondary s-0 rounded-pill px-2">
                                                            <i className="fa fa-cloud-upload mr-2 text-muted"/>
                                                            <small className="text-uppercase font-weight-bold text-muted">Browse</small></label>
                                                    </div>
                                                </div>

                                                <div className="image-area mt-4">
                                                    <img id="imageResult" src={this.state.fileURL} alt=""
                                                        className="img-fluid rounded shadow-sm mx-auto d-block"/>
                                                </div>
                                            </div>
                                        </div>
                                        <LoadingButton path="/#/recommendation" text="Send Image" state={this.state}/>

                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>


            </div>
        );
    }
}

export default UploadImage
