import React from 'react';
import {Card, Button} from 'react-bootstrap'
import ReactCardFlip from 'react-card-flip'
import QRCode from 'qrcode.react'
import Measure from 'react-measure'

import '../App.css';
//<Card.Header className='text-center text-white mb-3 mt-3 '>{props.title}</Card.Header>
class SCard extends React.Component {
    constructor() {
        super();
        this.state = {
            isFlipped: false,
            imgDims: {
                width: -1,
                heigth: -1
            }
        };
        this.handleClick = this.handleClick.bind(this);
    }

    handleClick(e) {
        e.preventDefault();
        this.setState(prevState => ({ isFlipped: !prevState.isFlipped }));
    }

    download(e) {
        //alert(e)
        //console.log(e.target.href);
        let props = this.props
        fetch(props.src, {
            method: "GET",
            headers: {}
        })
            .then(response => {
                response.arrayBuffer().then(function(buffer) {
                    //const url = window.URL.createObjectURL(new Blob([buffer]));
                    const link = document.createElement("a");
                    link.href = props.src;
                    link.setAttribute("download", 'image.png'); //or any other extension
                    document.body.appendChild(link);
                    link.click();
                });
            })
            .catch(err => {
                console.log(err);
            });
    };

    handleSelection = () => {
        this.props.onSelectStyle(this.props.id)

    }

    render() {
        let button = null;
        if (this.props.hairstyle){
            button =
                <Button variant="secondary" size="md" onClick={e => this.handleSelection()}>
                    Try me!
                </Button>
        }else{
            button =
                <>
                    <Button variant="secondary" size="md" onClick={e => this.download(this.props.src)}>
                        Download
                    </Button>
                    <Button variant="secondary" size="md" onClick={this.handleClick}>
                        QR Code
                    </Button>
                </>
        }
        const { width, height } = this.state.imgDims
        return (
            <ReactCardFlip
                isFlipped={this.state.isFlipped}
                flipSpeedBackToFront={0.3}
                flipSpeedFrontToBack={0.3}>
                <Card className="scard">
                    <Measure
                        bounds
                        onResize={contentRect => {
                            this.setState({ imgDims: contentRect.bounds })
                        }}>
                        {({measureRef}) => <Card.Img ref={measureRef} className="scard-image" variant="top" src={this.props.src}/>}
                    </Measure>
                    <Card.Footer className="text-muted mt-4">
                        <div className="d-grid gap-2">
                            {button}
                        </div>
                    </Card.Footer>
                </Card>
                <Card className="scard">
                    <div style={{width: width, height: height}}>
                        <QRCode value="https://youtu.be/dQw4w9WgXcQ" className="qr-code" size="auto" renderAs="svg" />
                    </div>
                    <Card.Footer className="text-muted mt-4">
                        <div className="d-grid gap-2">
                            {button}
                        </div>
                    </Card.Footer>
                </Card>
            </ReactCardFlip>
        )
    }
}

export default SCard
