from django.http import JsonResponse


def test(request):
    data = {
        'mensaje': 'Prueba de API'
    }
    return JsonResponse(data)
