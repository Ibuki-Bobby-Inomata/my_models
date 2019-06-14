var http = require('http');
var url = require("url");
var fs = require('fs');
var querystring = require('querystring');

var server = http.createServer(function(req, res) {
  var pathname = url.parse(req.url).pathname;

  switch (pathname) {

    case '/':
      if (req.method === 'GET') {
        fs.readFile('./index.html', 'UTF-8', function(err, data){
          res.writeHead(200, {'Content-Type': 'text/html'});
          res.end(data);
        });
      }
    break;

    case '/result':
      if (req.method === 'POST') {
        var body = '';
        req.on('data', function(chunk) {
          body += chunk;
        });

        req.on('end',function(){
          var form = querystring.parse(body);
          var text = form.result;
            fs.readFile('./index.html', 'UTF-8', function(err, data){
                res.writeHead(200, {'Content-Type': 'text/html'});
                res.end('「'+ text + '」と入力しました。');
            });
        });
      }
    break;

  }
});

server.listen(3000, function() {
    console.log('listening on port 3000');
});