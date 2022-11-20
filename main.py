from flask import Flask, Blueprint
from flask_cors import CORS

from routes.dqr_routes import navigator_dqr
from routes.aa_routes import navigator_aa
from routes.mb_routes import navigator_mb
from routes.sf_routes import navigator_sf


def config():
    app = Flask(__name__)
    CORS(app)
    builder_routes_path(app, navigator_dqr, 'dqr_routes', '/dqr')
    builder_routes_path(app, navigator_aa, 'aa_routes', '/aa')
    builder_routes_path(app, navigator_mb, 'mb_routes', '/mb')
    builder_routes_path(app, navigator_sf, 'sf_routes', '/sf')
    return app


def builder_routes_path(app, navigator, route, url_prefix):
    router_build = Blueprint(route, __name__)
    navigator(router_build)
    app.register_blueprint(router_build, url_prefix=url_prefix)


if __name__ == '__main__':
    config().run(debug=True)
