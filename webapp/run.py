from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from dash_app import create_dashboard
mydash = create_dashboard()

from flask_app import create_app
myflask = create_app()

application = DispatcherMiddleware(myflask, 
                           {'/dashboard': mydash.server})

if __name__ == '__main__':
    run_simple('localhost', 8080, application, use_reloader=True, threaded=True)
