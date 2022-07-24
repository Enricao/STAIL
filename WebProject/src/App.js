import './App.css';

import Navigation from './Components/Navbar'
import {Route, Switch} from 'react-router-dom'

import Main from './Pages/Main'
import UploadImage from './Pages/UploadImage'
import Recommendation from './Pages/Recommendation'
import Global from './Components/Global'
import React from "react";

//  "react-scripts start"
function App () {

        return(

            <div>
                <Navigation/>

                <div className="App-display">

                    <Switch>
                        <Route path='/' exact={true} component={Main}/>
                        <Global>
                            <Route path='/uploadImage' component={UploadImage} />
                            <Route path='/recommendation' component={Recommendation}/>
                        </Global>
                    </Switch>


                </div>

            </div>
        )
}

export default App;