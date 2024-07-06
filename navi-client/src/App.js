// import React, { useEffect, useState, Button } from "react";

// function App() {
//   const button = () => {
//     fetch("http://localhost:5000/ask")
//       .then((response) => response.json())
//       .then((data) => setData(data));
//   };
//   const [data, setData] = useState(null);

//   // useEffect(() => {
//   //   fetch("http://localhost:5000/api/data")
//   //     .then((response) => response.json())
//   //     .then((data) => setData(data));
//   // }, []);

//   return (
//     <div className="App">
//       <p>임시 텍스트</p>

//       <button onClick={button}>버튼임</button>
//       <p>임시 텍스트4</p>
//       <header className="App-header">{data ? <p>{data}</p> : <p>Loading...</p>}</header>
//     </div>
//   );
// }

// export default App;

import React, { useState } from "react";

function App() {
  const [data, setData] = useState(null);

  const handleClick = () => {
    fetch("http://localhost:5000/ask")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        setData(data.data); // 서버에서 전송한 데이터를 설정
      })
      .catch((error) => {
        console.error("There has been a problem with your fetch operation:", error);
      });
  };

  return (
    <div className="App">
      <p>임시 텍스트</p>
      <button onClick={handleClick}>버튼임</button>
      <p>임시 텍스트4</p>
      <header className="App-header">{data ? <p>{data}</p> : <p>Loading...</p>}</header>
    </div>
  );
}

export default App;
