import logo from '../Resources/pisc-1.svg';
import Button from 'react-bootstrap/Button'

function Main() {
    return (
        <div className="App-welcome">
            <div className="home-center">
                <div className="home-desc-center">

                    <div className="container">
                        <div className="align-items-center row">
                            <div className="col-sm-6">
                                <div className="mt-20 home-2-content"><h1
                                    className="text-white font-weight-normal home-2-title display-4 mb-0">Welcome to
                                    Stail</h1>
                                    <p className="text-white-70 mt-4 f-15 mb-0">Where you can show us a picture, and DL
                                        algorithms will tell what suits you best (stylishly)</p>
                                    <div className="mt-5">

                                        <Button
                                            variant="secondary" href="/#/uploadImage"> Get started
                                        </Button>

                                    </div>
                                </div>
                            </div>
                            <div className="col-md-4">
                                <div className="mt-40 home-2-content position-relative"><img src={logo} alt="logo"
                                                                                             className="App-logo img-fluid mx-auto d-block home-2-img mover-img"
                                                                                             width="300"/>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    );
}

export default Main