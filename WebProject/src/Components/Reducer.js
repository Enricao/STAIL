const Reducer = (state, action) => {
    switch (action.type) {
        case 'RECOMMENDATION':
            return {
                ...state,
                posts: state.posts.concat(action.payload)
            };
        default:
            return state;
    }
};

export default Reducer;