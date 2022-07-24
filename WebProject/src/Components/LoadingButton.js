import Button from 'react-bootstrap/Button'
import React, {useState, useEffect} from 'react';


const sleep = (milliseconds) => {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
}

function LoadingButton(props) {
    const [isLoading, setLoading] = useState(false);

    useEffect( () => {
        async function fetchRecommendation() {
            console.log('uploaded image')
            let formData = new FormData()
            formData.append('image', props.state.file) //'/api/recommendation'
            await fetch('/api/recommendation', {
                method: 'POST',
                body: formData,
            }).then(response => {
                response.json()
                    .then(r =>
                        //dispatch({type: 'RECOMMENDATION', payload: r.image})
                       storeImages(r)
                    )
            }).catch(error => console.log(error));
        }

        function storeImages(r){
            localStorage.setItem('user', JSON.stringify(r.user))
            console.log(r.image)
            localStorage.setItem('recommendation', JSON.stringify(r.image))
            console.log(r.user)

        }

        async function callFetchRecommendation() {
            await fetchRecommendation()
        }

        async function storeUserImage() {
            localStorage.setItem('user', JSON.stringify())
        }

        if (isLoading) {
            if (props.state.file) {
                callFetchRecommendation()
                sleep(1200)
                window.location.href = props.path
                setLoading(false)
            } else {
                setLoading(false)
            }

        }
    }, [isLoading]);


    const checkGlobal = () => console.log((JSON.parse(localStorage.getItem('recommendation'))))
    const clear = () => localStorage.clear()
    const recomm = () => window.location.href = props.path

    const handleClick = () => setLoading(true);
    const clickDone = () => {
        window.location.href = props.path;
        setLoading(false);
    }


    return (
        <div className="col-lg-6 mx-auto">
            <div className="d-grid gap-2">
                <Button
                    variant="secondary"
                    disabled={isLoading}
                    onClick={!isLoading ? handleClick : null}>
                    {isLoading ? 'Loading…' : props.text}
                </Button>
            </div>
        </div>

    );
}

//ReactDOM.render(<LoadingButton/>);

export default LoadingButton