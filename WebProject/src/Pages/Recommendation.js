import SCard from '../Components/SCard'
import {Col, Row} from 'react-bootstrap'

import React, {useEffect, useState} from 'react';

//const sharp = require('sharp');
const sleep = (milliseconds) => {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
}


async function fetchOutput(ids) {
    let arr = [];
    for (var i = 0; i <= ids.length; i++) {
        await fetch('/api/recommendation/'.concat(String(parseInt(ids[i]))), {
            method: 'GET',
        }).then(response => {
            response.json().then(r => arr.push(r.hairstyle))
        }).catch(error => console.log(error));
    }
    return arr
}

async function fetchUser(id) {
    let user = [];
    await fetch('/api/recommendation/user/'.concat(String(parseInt(id))), {
        method: 'GET',
    }).then(response => {
        response.json().then(r => user.push(r.user))
    }).catch(error => console.log(error));
    return user
}

async function fetchFusion(style, user) {
    let fusion = [];
    await fetch('/api/recommendation/'.concat(String(user), '/fuse/', String(style)), {
        method: 'GET',
    }).then(response => {
        response.json().then(r => fusion.push(r.fuse))
    }).catch(error => console.log(error));
    return fusion
}

async function fetchUserId(){
    return await JSON.parse(localStorage.getItem('user'))
}

async function fetchStyleId(){
    return await JSON.parse(localStorage.getItem('recommendation'))

}


function Recommendation() {
    //console.clear()
    const [hairstyles, setHairstyles] = useState([]);
    const [user, setUser] = useState(null);
    const [fuse, setFuse] = useState(null);
    const [src, setSrc] = useState(null);
    const [selectedStyle, setStyle] = useState(null);
    //const [user_id, setUserId] = useState(null);
    //const [hairstyle_ids, setStylesId] = useState(null);
    const hairstyle_ids = JSON.parse(localStorage.getItem('recommendation'))
    const user_id = JSON.parse(localStorage.getItem('user'))

    const[isBusy, setBusy] = useState(true);
    const[isChange, setChange] = useState(false);

    handleStyleSelection = handleStyleSelection.bind(this)

    console.log('')
    console.log('---------------------Re-rendering-----------------')
    console.log('Hairstyle ids:', hairstyle_ids)
    console.log('User id', user_id)
    console.log('User image',user)
    console.log('Hairstyle images', hairstyles)
    console.log('Hairstyle src', src)
    console.log('Hairstyle Fuse', fuse)
    console.log('----------------- Finish Re-rendering-----------------')
    console.log('')

    useEffect(() => {
        if(isBusy) {
            //setUserId(fetchUserId().then(data => setUserId(data)))
            //setStylesId(fetchStyleId().then(data => setStylesId(data)))
            if(user_id != null && hairstyle_ids != null) {
                fetchOutput(hairstyle_ids).then(data => setHairstyles(data))
                fetchUser(user_id).then(data => setUser(data))
                fetchUser(user_id).then(data => setSrc(data))
                setBusy(false)
            }
        }
    },[hairstyle_ids, isBusy, user_id])

    useEffect(() => {
        if(isChange) {
            console.log('I am here')
            fetchFusion(selectedStyle, user_id).then(data => setSrc(data))
            fetchFusion(selectedStyle, user_id).then(data => setFuse(data))
            console.log('This is the src after selection',src)
            setChange(false)
        }
    },[isChange])

    function handleStyleSelection(style) {
        setStyle(style)
        setChange(true)
    }

    return (
        <div className="container py-5">

            <header className="text-white text-center">
                <h1 className="display-4">This is you</h1>
            </header>
            <hr/>
            <Row className="g-4 d-flex justify-content-center">

                <Col md={{span: 3}}>
                    <SCard src={`data:image/png;base64,${src}`}>
                    </SCard>
                </Col>

            </Row>
            <hr/>

            <header className="text-white text-center">
                <h1 className="display-4">These are your recommendations</h1>
                <p className="lead mb-4">Hope they suit you ;)</p>
            </header>

            <Row s={1} md={3} className="g-4 d-flex justify-content-center">
                {Array.from({length: 3}).map((_, idx) => (
                    <Col md={{span: 3}}>
                        <SCard id={hairstyle_ids[idx]}
                               src={`data:image/png;base64,${hairstyles[idx]}`} hairstyle={true}
                               onSelectStyle={handleStyleSelection}>
                        </SCard>
                    </Col>
                ))}
            </Row>
        </div>
    );
}

export default Recommendation