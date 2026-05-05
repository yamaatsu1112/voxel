use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{Attribute, Expr, Ident, Token, Visibility};

struct DefineSamplerInput {
    attrs: Vec<Attribute>,
    vis: Visibility,
    name: Ident,
    config: Expr,
}

impl Parse for DefineSamplerInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![struct]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![;]>()?;

        // config = <expr>;
        let ident: Ident = input.parse()?;
        assert_eq!(ident, "config", "Expected 'config'");
        input.parse::<Token![=]>()?;
        let config: Expr = input.parse()?;
        input.parse::<Token![;]>()?;

        Ok(DefineSamplerInput {
            attrs,
            vis,
            name,
            config,
        })
    }
}

pub fn define_sampler_impl(input: TokenStream) -> TokenStream {
    let DefineSamplerInput {
        attrs,
        vis,
        name,
        config,
    } = syn::parse_macro_input!(input as DefineSamplerInput);

    let expanded = quote! {
        #(#attrs)*
        #vis struct #name;

        impl crate::vulkan::resource_config::SamplerMarker for #name {
            const CONFIG: crate::vulkan::resource_config::SamplerConfig = #config;

            fn id() -> &'static std::sync::atomic::AtomicUsize {
                // SAFETY: AtomicUsize and usize have identical memory layout.
                static ID: std::sync::atomic::AtomicUsize =
                    unsafe { std::mem::transmute(usize::MAX) };
                &ID
            }
        }
    };

    expanded.into()
}
