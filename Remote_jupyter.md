## A tourial of accessing jupyter_lab in a remote serve

1. Download and install jupyter/lab in your serve
2. In ur own PC/Mac, run ssh -L(the port u want to use in ur device, e.g. 8888):(the port jupyter use in ur server, e.g. localhost:8888) id@ip_address -i key(if have key)
3. Run jupyter at server, get token and open it in browser(may not exist token, it's normal, just open the url instead)
4. May need u to input password, just press "Enter"(if there is no token, it will need password)
5. Done!
